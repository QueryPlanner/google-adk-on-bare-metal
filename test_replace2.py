import re

ORIGINAL_PACKAGE_NAME = "agent"
ORIGINAL_REPO_NAME = "google-adk-on-bare-metal"
ORIGINAL_GITHUB_OWNER = "QueryPlanner"

github_owner = "my-user"
repo_name = "my-queryplanner-bot"
package_name = "my_queryplanner_bot"

replacements = {
    f"https://github.com/{ORIGINAL_GITHUB_OWNER}/{ORIGINAL_REPO_NAME}/": f"https://github.com/{github_owner}/{repo_name}/",
    ORIGINAL_PACKAGE_NAME: package_name,
    ORIGINAL_REPO_NAME: repo_name,
    ORIGINAL_GITHUB_OWNER.lower(): github_owner.lower(),
}

text = f"Visit https://github.com/{ORIGINAL_GITHUB_OWNER}/{ORIGINAL_REPO_NAME}/\nPackage: {ORIGINAL_PACKAGE_NAME}\nRepo: {ORIGINAL_REPO_NAME}\nOwner: {ORIGINAL_GITHUB_OWNER.lower()}"

modified = text
for old, new in replacements.items():
    modified = modified.replace(old, new)

print("--- ORIGINAL ---")
print(text)
print("--- MODIFIED ---")
print(modified)
