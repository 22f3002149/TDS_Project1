## Static hosting: GitHub Pages

[GitHub Pages](https://pages.github.com/) is a free hosting service that turns your GitHub repository directly into a static website whenever you push it. This is useful for sharing analysis results, data science portfolios, project documentation, and more.

Common Operations:

```bash
# Create a new GitHub repo
mkdir my-site
cd my-site
git init

# Add your static content
echo "<h1>My Site</h1>" > index.html

# Push to GitHub
git add .
git commit -m "feat(pages): initial commit"
git push origin main

# Enable GitHub Pages from the main branch on the repo settings page
```

Best Practices:

1. **Keep it small**
   - [Optimize images](https://developer.mozilla.org/en-US/docs/Learn_web_development/Extensions/Performance/Multimedia). Prefer SVG over WEBP over 8-bit PNG.
   - [Preload](https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/rel/preload) critical assets like stylesheets
   - Avoid committing large files like datasets, videos, etc. directly. Explore [Git LFS](https://git-lfs.github.com/) instead.

Tools:

- [GitHub Desktop](https://desktop.github.com/): GUI for Git operations
- [GitHub CLI](https://cli.github.com/): Command line interface
- [GitHub Actions](https://github.com/features/actions): Automation

[Host a website using GitHub Pages: Here\'s a detailed description of the image:\n\n**Overall Impression:** The image is a screenshot from a video tutorial or presentation about hosting websites using GitHub Pages. It features a woman speaking directly to the camera with a GitHub Pages interface visible on the left side of the screen. \n\n**Visual Elements:**\n\n* **Woman:** A woman with medium-length, curly brown hair is visible on the right side of the screen. She is wearing a light-colored blouse and appears to be smiling and speaking directly at the viewer.\n* **GitHub Pages Interface:** On the left side of the screen is the GitHub Pages interface. Text on the interface reads, "GitHub Pages," and a description saying "GitHub Pages is designed to host your personal, organization." There is also a "Theme Chooser" and options to select a theme.\n* **Text Overlay:** Large, bold text overlays the image reading "Hosting websites." The words "Hosting" and "websites" are separated and emphasized with a bright blue color.\n* **Color Palette:** The color scheme is based on dark shades with brighter blue and white elements, primarily from the text overlays and the GitHub interface.\n\n**Composition:** The composition is split into two main parts: the woman speaking on the right and the GitHub Pages interface on the left. The text overlays are strategically placed to draw attention to the topic.\n\n**Context:** Given the visual elements, the image likely originates from a tutorial or educational content explaining how to host websites using GitHub Pages.](https://youtube.com/shorts_WqOXxoGSpbs)

[Deploy your first GitHub Pages Website: Here\'s a detailed description of the image:\n\n**Overall Impression:**\n\nThe image displays a screenshot of a code editor (likely Visual Studio Code) displaying a file named "README.md" and a sidebar displaying GitHub Actions information. The overall theme is dark with a high-contrast color scheme.\n\n**Detailed Breakdown:**\n\n* **Code Editor Window:** The main portion of the screen shows the contents of a "README.md" file. The file contains code snippets, likely related to a GitHub Actions workflow. There are comments, variable definitions (like `ENV`, `APP_ID`, `CLIENT_SECRET`), and code blocks.\n* **GitHub Actions Sidebar:** On the left side, there\'s a sidebar labeled "GitHub Actions." It details workflow runs, statuses, and job progress.\n * There\'s a section titled "Current Branch" and "Workflows."\n * Under "Workflows," there’s a list of workflow runs, identified by numbers like #4404, #4403, #4402, etc. Some have a green checkmark indicating success, while others have a yellow or orange indicator implying pending or failed status.\n * The currently selected workflow run is "#4404" named "Invitation.label"\n* **Tabs:** Several tabs are open at the top of the editor window, including "Extension: GitHub Actions", "Preview README.md", and "Quokka: What\'s New".\n* **Bottom Bar:** The bottom of the screen displays status bar information, including the project name ("blackbytes-on-main-mrpwdf"), the branch ("main"), line numbers, and indicators for indentation, spaces, and line endings.\n* **Color Scheme:** The editor uses a dark background with bright, contrasting text and highlighting, typical of many dark themes.\n\n**In summary,** the image showcases a developer working with GitHub Actions within a code editor, likely troubleshooting or monitoring a workflow run. The interface is detailed, suggesting a technical context related to software development and version control.](https://youtu.be_sT_zXIX3ZA0)
