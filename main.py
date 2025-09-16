"""
PR Reviewer app
Single-file Flask app that provides a minimal frontend to submit a public GitHub Pull Request URL,
calls OpenAI to review the PR, posts review comments and labels the PR using a GitHub token.

Environment variables expected :
  - GITHUB_TOKEN : Personal access token with `repo` scope (for public repos this should work)
  - OPENAI_API_KEY : OpenAI API key (used by `openai` Python package)

Notes & limitations:
  - This is a minimal, opinionated starter. It posts a single review comment on the PR conversation
    and adds labels. It does not create inline file comments (those require file path + position mapping).
  - The app fetches up to 5 changed files and truncates each file to avoid huge prompts.
  - You may need to switch the OpenAI model name depending on your account. See the comment in `call_openai_review`.

Usage:
  pip install -r requirements.txt 
  export GITHUB_TOKEN="ghp_..."
  export OPENAI_API_KEY="sk-..."
  python pr_reviewer_app.py

Open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, request, render_template_string, jsonify
import os
import requests
import openai
import json
import re
import textwrap

app = Flask(__name__)

# Configuration
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

# Minimal HTML front-end template
HTML = """
<!doctype html>
<title>PR Reviewer</title>
<h2>Pull Request Reviewer</h2>
<form id="prForm" method="post" action="/review">
  <label for="pr_url">Public GitHub PR URL:</label><br>
  <input type="url" id="pr_url" name="pr_url" size="80" required placeholder="https://github.com/owner/repo/pull/123"><br><br>
  <label for="token">(optional) GitHub token (or set GITHUB_TOKEN env var):</label><br>
  <input type="password" id="token" name="token" size="80" placeholder="ghp_xxx"><br><br>
  <button type="submit">Review PR</button>
</form>
<pre id="result"></pre>
<script>
const form = document.getElementById('prForm');
const result = document.getElementById('result');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  result.textContent = 'Sending...';
  const data = new FormData(form);
  const resp = await fetch('/review', { method: 'POST', body: data });
  const js = await resp.json();
  if (resp.ok) {
    result.textContent = JSON.stringify(js, null, 2);
  } else {
    result.textContent = 'Error: ' + JSON.stringify(js, null, 2);
  }
});
</script>
"""

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
        selected.append({'filename': filename, 'status': f.get('status'), 'changes': f.get('changes'), 'contents': file_contents})
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
        'content': 'You are a senior software engineer and security expert. Produce a clear, concise review of the pull request. Return output as JSON with keys: summary (short), comments (list of {path,comment}), labels (list of strings). Do not add any extra text outside the JSON.'
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
        user_text += f"\n---\nFile: {f['filename']} (status: {f['status']}, changes: {f['changes']})\n{f['contents']}\n"

    user_text += "\nInstructions:\n- Review for correctness, code logic, security issues, architecture/design concerns, and potential build/test problems.\n- Produce up to 6 actionable comments.\n- Suggest labels chosen from: security-issue, code-logic-issue, build-issue, docs-issue, style-issue, perf-issue, other.\n- Output ONLY valid JSON.\n"

    prompt.append({'role': 'user', 'content': user_text})
    return prompt


from openai import OpenAI

client = OpenAI(api_key=OPENAI_KEY)

def call_openai_review(prompt_messages, model="gpt-4o-mini"):
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    resp = client.chat.completions.create(
        model=model,
        messages=prompt_messages,
        max_tokens=1000,
        temperature=0.0,
    )
    return resp.choices[0].message.content

def parse_model_json_safe(text):
    # Models sometimes wrap JSON in ``` or extraneous text. Try to extract the first JSON object.
    # Find first occurrence of '{' and last '}' and attempt to parse.
    try:
        start = text.index('{')
        end = text.rfind('}')
        json_text = text[start:end+1]
        return json.loads(json_text)
    except Exception as e:
        # fallback: try direct json.loads
        try:
            return json.loads(text)
        except Exception:
            # As a last resort return a best-effort structure
            return {'summary': text.strip(), 'comments': [], 'labels': []}


def post_review_and_labels(owner, repo, pr_number, token, review_json):
    # Create a PR review (single comment) summarizing findings
    summary = review_json.get('summary') or ''
    comments = review_json.get('comments') or []

    body = f"Automated review summary:\n\n{summary}\n\nDetailed comments:\n"
    for c in comments:
        path = c.get('path', '<unknown>')
        comment = c.get('comment') or c.get('message') or ''
        body += f"\n- {path}: {comment}\n"

    # Post the review on the PR as a review event
    review_payload = {
        'body': body,
        'event': 'COMMENT'
    }
    review_resp = github_post(f'/repos/{owner}/{repo}/pulls/{pr_number}/reviews', review_payload, token=token)

    # Add labels on the issue associated with the PR
    labels = review_json.get('labels') or []
    # sanitize labels to known mapping
    allowed = {'security-issue','code-logic-issue','build-issue','docs-issue','style-issue','perf-issue','other'}
    to_add = [l for l in labels if l in allowed]
    if to_add:
        labels_resp = github_post(f'/repos/{owner}/{repo}/issues/{pr_number}/labels', {'labels': to_add}, token=token)
    else:
        labels_resp = None

    return {'review': review_resp, 'labels': labels_resp}


# Routes
@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/review', methods=['POST'])
def review():
    pr_url = request.form.get('pr_url')
    token = request.form.get('token') or GITHUB_TOKEN
    if not pr_url:
        return jsonify({'error': 'pr_url is required'}), 400
    parsed = parse_pr_url(pr_url)
    if not parsed:
        return jsonify({'error': 'Invalid GitHub PR URL. Expected: https://github.com/owner/repo/pull/123'}), 400
    owner, repo, pr_number = parsed

    if not token:
        return jsonify({'error': 'No GitHub token provided. Set GITHUB_TOKEN env var or provide token in form.'}), 400

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
        return jsonify({'error': 'Failed to post review/labels to GitHub', 'details': str(e), 'model_output': review_json}), 500

    return jsonify({'status': 'success', 'model_output': review_json, 'github_response': gh_resp})


if __name__ == '__main__':
    app.run(debug=True)