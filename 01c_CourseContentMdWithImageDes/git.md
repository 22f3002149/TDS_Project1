## Version Control: Git, GitHub

[Git](https://git-scm.com/) is the de facto standard for version control of software (and sometimes, data as well). It's a system that keeps track of changes you make to files and folders. It allows you to revert to a previous state, compare changes, etc. It's a central tool in any developer's workflow.

[GitHub](https://github.com/) is the most popular hosting service for Git repositories. It's a website that shows your code, allows you to collaborate with others, and provides many useful tools for developers.

Watch these introductory videos to learn the basics of Git and GitHub (98 min):

[Git Tutorial for Beginners: Command-Line Fundamentals (30 min): Here\'s a point-by-point description of the image:\n\n* **Subject:** The image features a graphic related to "git", a popular version control system.\n* **Logo:** A red octopus-like symbol representing git is positioned above the text.\n* **Text:** The text "git" is displayed in a bold, white font.\n* **Subtitle:** Below "git" are the words "Command-Line Fundamentals" in a larger, bold, white font.\n* **Background:** The background is a muted, grayish-brown color.\n* **Overall:** It appears to be promotional or informational material, likely for a course or tutorial about using git through the command line.](https://youtu.be_HVsySz-h9r4)

[Git and GitHub for Beginners - Crash Course (68 min): Here\'s a detailed description of the image:\n\n* **Overall Impression:** The image is a title card or thumbnail promoting a "Git and GitHub Crash Course."\n* **Background:** The background is a solid dark blue/black color.\n* **Logos:** Three logos are present at the top of the image:\n * A compass with a droplet-like shape within it, representing Bitbucket.\n * A flame symbol, representing GitLab.\n * An octocat symbol, representing GitHub.\n* **Text:** Below the logos, the text "Git and GitHub" is prominently displayed in a large, bold, white font. Below that, in a slightly smaller, yet still bold white font, is the text "Crash Course."\n* **Style:** The image has a clean, minimalist design with a focus on clear typography and brand recognition through the logos.\n* **Purpose:** It’s likely used to advertise or introduce a tutorial or course related to the Git version control system and the GitHub platform.\n\n\n\n](https://youtu.be/RGOj5yH7evk)

Essential Git Commands:

```bash
# Repository Setup
git init                   # Create new repo
git clone url              # Clone existing repo
git remote add origin url  # Connect to remote

# Basic Workflow
git status                 # Check status
git add .                  # Stage all changes
git commit -m "message"    # Commit changes
git push origin main       # Push to remote

# Branching
git branch                 # List branches
git checkout -b feature    # Create/switch branch
git merge feature          # Merge branch
git rebase main            # Rebase on main

# History
git log --oneline          # View history
git diff commit1 commit2   # Compare commits
git blame file             # Show who changed what
```

Best Practices:

1. **Commit Messages**

   ```bash
   # Good commit message format
   type(scope): summary

   Detailed description of changes.

   # Examples
   feat(api): add user authentication
   fix(db): handle null values in query
   ```

2. **Branching Strategy**

   - main: Production code
   - develop: Integration branch
   - feature/\*: New features
   - hotfix/\*: Emergency fixes

3. **Code Review**
   - Keep PRs small (<400 lines)
   - Use draft PRs for WIP
   - Review your own code first
   - Respond to all comments

Essential Tools

- [GitHub Desktop](https://desktop.github.com/): GUI client
- [GitLens](https://gitlens.amod.io/): VS Code extension
- [gh](https://cli.github.com/): GitHub CLI
- [pre-commit](https://pre-commit.com/): Git hooks
