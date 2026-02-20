
ORIGINAL_PACKAGE_NAME = "agent"
ORIGINAL_REPO_NAME = "google-adk-on-bare-metal"
ORIGINAL_GITHUB_OWNER = "QueryPlanner"

github_owner = "my-user"
repo_name = "my-queryplanner-bot"
package_name = "my_queryplanner_bot"

config_github_owner = github_owner
config_repo_name = repo_name
config_package_name = package_name

replacements = {
    f"https://github.com/{ORIGINAL_GITHUB_OWNER}/{ORIGINAL_REPO_NAME}/": "<URL_PLACEHOLDER>",  # noqa: E501
    ORIGINAL_PACKAGE_NAME: "<PKG_PLACEHOLDER>",
    ORIGINAL_REPO_NAME: "<REPO_PLACEHOLDER>",
    ORIGINAL_GITHUB_OWNER.lower(): "<OWNER_PLACEHOLDER>",
    "<URL_PLACEHOLDER>": f"https://github.com/{config_github_owner}/{config_repo_name}/",
    "<PKG_PLACEHOLDER>": config_package_name,
    "<REPO_PLACEHOLDER>": config_repo_name,
    "<OWNER_PLACEHOLDER>": config_github_owner.lower(),
}

text = f"Visit https://github.com/{ORIGINAL_GITHUB_OWNER}/{ORIGINAL_REPO_NAME}/\nPackage: {ORIGINAL_PACKAGE_NAME}\nRepo: {ORIGINAL_REPO_NAME}\nOwner: {ORIGINAL_GITHUB_OWNER.lower()}"  # noqa: E501

modified = text
for old, new in replacements.items():
    modified = modified.replace(old, new)

print("--- ORIGINAL ---")
print(text)
print("--- MODIFIED ---")
print(modified)
