

from flask import Flask, render_template, request, jsonify
import os
import requests
import json
import re

from openai import OpenAI

app = Flask(__name__)

# Configuration
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# Helpers
PR_URL_REGEX = re.compile(r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)")


def parse_pr_url(url):
    m = PR_URL_REGEX.search(url.strip())
    if not m:
        return None
    return m.group('owner'), m.group('repo'), int(m.group('number'))


def github_get(path, token=None, params=None):
    headers = {'Accept': 'application/vnd.github+json'}
    if token:
        headers['Authorization'] = f'token {token}'
    url = f'https://api.github.com{path}'
    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    return r.json()


def github_post(path, json_data, token=None):
    headers = {'Accept': 'application/vnd.github+json'}
    if token:
        headers['Authorization'] = f'token {token}'
    url = f'https://api.github.com{path}'
    r = requests.post(url, headers=headers, json=json_data)
    r.raise_for_status()
    return r.json()


def fetch_pr_and_files(owner, repo, pr_number, token=None, max_files=5, max_chars_per_file=2000):
    pr = github_get(f'/repos/{owner}/{repo}/pulls/{pr_number}', token=token)
    files = github_get(f'/repos/{owner}/{repo}/pulls/{pr_number}/files', token=token)
    selected = []
    for f in files[:max_files]:
        filename = f.get('filename')
        raw_url = f.get('raw_url')
        try:
            file_contents = requests.get(raw_url).text
        except Exception:
            file_contents = '<<failed to fetch file contents>>'
        if len(file_contents) > max_chars_per_file:
            file_contents = file_contents[:max_chars_per_file] + '\n\n...TRUNCATED...'
        selected.append({
            'filename': filename,
            'status': f.get('status'),
            'changes': f.get('changes'),
            'contents': file_contents
        })
    return pr, selected


def build_prompt(pr, files):
    title = pr.get('title', '')
    body = pr.get('body') or ''
    author = pr.get('user', {}).get('login')
    branch = pr.get('head', {}).get('ref')
    base = pr.get('base', {}).get('ref')

    prompt = []
    prompt.append({
        'role': 'system',
        'content': (
            "You are a senior software engineer and security expert. "
            "Produce a clear, concise review of the pull request. "
            "Return output strictly as JSON with keys: \n"
            "  - summary (short text summary of the PR quality),\n"
            "  - comments (list of objects with {path, block, issue, corrected_code, comment}),\n"
            "  - labels (list of strings).\n\n"
            "For each comment:\n"
            "  - path: filename where the issue occurs.\n"
            "  - block: the specific code block (few lines) where the issue exists.\n"
            "  - issue: concise description of the problem.\n"
            "  - corrected_code: corrected code snippet that fixes the issue.\n"
            "  - comment: a clear explanation of the issue and how the fix resolves it.\n\n"
            "Important rules:\n"
            "- Output ONLY valid JSON (no extra text).\n"
            "- Maximum 6 comments.\n"
            "- Labels must be chosen from: security-issue, code-logic-issue, build-issue, "
            "docs-issue, style-issue, perf-issue, other."
            #test
        )
    })

    user_text = f"""
PR title: {title}
Author: {author}
Branch: {branch} -> {base}

Description:
{body}

Changed files (up to {len(files)}):
"""
    for f in files:
        user_text += (
            f"\n---\nFile: {f['filename']} (status: {f['status']}, changes: {f['changes']})\n"
            f"{f['contents']}\n"
        )

    user_text += "\nInstructions:\n- Review for correctness, code logic, security issues, architecture/design concerns, and potential build/test problems.\n- Produce up to 6 actionable comments.\n- Suggest labels chosen from: security-issue, code-logic-issue, build-issue, docs-issue, style-issue, perf-issue, other.\n- Output ONLY valid JSON.\n"

    prompt.append({'role': 'user', 'content': user_text})
    return prompt


def call_openai_review(prompt_messages, model="gpt-4o-mini"):
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    resp = client.chat.completions.create(
        model=model,
        messages=prompt_messages,
        max_tokens=1200,
        temperature=0.0,
    )
    return resp.choices[0].message.content


def parse_model_json_safe(text):
    try:
        start = text.index('{')
        end = text.rfind('}')
        json_text = text[start:end+1]
        return json.loads(json_text)
    except Exception:
        try:
            return json.loads(text)
        except Exception:
            return {'summary': text.strip(), 'comments': [], 'labels': []}


def post_review_and_labels(owner, repo, pr_number, token, review_json):
    summary = review_json.get('summary') or ''
    comments = review_json.get('comments') or []

    body = f"### 🤖 Automated Review Summary\n\n{summary}\n\n### 📝 Detailed Comments:\n"
    for c in comments:
        path = c.get('path', '<unknown>')
        block = c.get('block', '')
        issue = c.get('issue', '')
        corrected = c.get('corrected_code', '')
        explanation = c.get('comment', '')

        body += f"\n**File:** `{path}`\n"
        if issue:
            body += f"- **Issue:** {issue}\n"
        if block:
            body += f"- **Problematic Code:**\n```python\n{block}\n```\n"
        if corrected:
            body += f"- **Suggested Fix:**\n```python\n{corrected}\n```\n"
        if explanation:
            body += f"- **Explanation:** {explanation}\n"

    review_payload = {
        'body': body,
        'event': 'COMMENT'
    }
    review_resp = github_post(
        f'/repos/{owner}/{repo}/pulls/{pr_number}/reviews',
        review_payload,
        token=token
    )

    labels = review_json.get('labels') or []
    allowed = {
        'security-issue', 'code-logic-issue', 'build-issue',
        'docs-issue', 'style-issue', 'perf-issue', 'other'
    }
    to_add = [l for l in labels if l in allowed]
    labels_resp = None
    if to_add:
        labels_resp = github_post(
            f'/repos/{owner}/{repo}/issues/{pr_number}/labels',
            {'labels': to_add},
            token=token
        )

    return {'review': review_resp, 'labels': labels_resp}


# Routes
@app.route('/')
def index():
    return render_template("ui.html")


@app.route('/review', methods=['POST'])
def review():
    pr_url = request.form.get('pr_url')
    token = request.form.get('token') or GITHUB_TOKEN
    if not pr_url:
        return jsonify({'error': 'pr_url is required'}), 400

    parsed = parse_pr_url(pr_url)
    if not parsed:
        return jsonify({
            'error': 'Invalid GitHub PR URL. Expected: https://github.com/owner/repo/pull/123'
        }), 400

    owner, repo, pr_number = parsed

    if not token:
        return jsonify({
            'error': 'No GitHub token provided. Set GITHUB_TOKEN env var or provide token in form.'
        }), 400

    try:
        pr, files = fetch_pr_and_files(owner, repo, pr_number, token=token)
    except requests.HTTPError as e:
        return jsonify({'error': 'Failed to fetch PR from GitHub', 'details': str(e)}), 400

    prompt = build_prompt(pr, files)
    try:
        model_text = call_openai_review(prompt)
    except Exception as e:
        return jsonify({'error': 'OpenAI review failed', 'details': str(e)}), 500

    review_json = parse_model_json_safe(model_text)

    try:
        gh_resp = post_review_and_labels(owner, repo, pr_number, token, review_json)
    except requests.HTTPError as e:
        return jsonify({
            'error': 'Failed to post review/labels to GitHub',
            'details': str(e),
            'model_output': review_json
        }), 500

    return jsonify({'status': 'success', 'model_output': review_json, 'github_response': gh_resp})


if __name__ == '__main__':
    app.run(debug=True)
