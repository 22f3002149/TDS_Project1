## LLM Video Screen-Scraping

Video screen-scraping with LLMs is a powerful technique for extracting structured data from screen recordings. This approach works with any visible screen content and bypasses traditional web scraping limitations like authentication or anti-scraping measures.

[Screen Scraping with Gemini: Here\'s a detailed description of the image:\n\n**Overall Impression:** The image shows a computer screen displaying the Google AI Studio interface alongside a video call window. The overall aesthetic is dark mode, with a focus on text and coding elements.\n\n**Key Elements:**\n\n* **Google AI Studio Interface:** A large portion of the screen is taken up by the Google AI Studio web application.\n * The interface is dark-themed, with a sidebar menu on the left.\n * The sidebar includes options like "AI Studio", "Create new prompt", "New text model", "View all models", etc.\n * A central pane displays a text box where a prompt is likely entered. It contains text about system instructions.\n * There’s a panel with "Run settings" including "Model", "Temperature", "Top P", "Top K", etc.\n* **Video Call Window:** On the right side of the screen is a video call window.\n * It features a person speaking, who appears to be a man with dark hair.\n * A label “Anand Subramanian” is visible beneath the video.\n* **Browser Tabs:** Multiple browser tabs are open at the top of the screen, suggesting that the user is engaged in multiple tasks.\n* **Operating System:** The image suggests the operating system is Windows, judging from the taskbar at the bottom.\n* **Application:** Google Chrome is used as the web browser.\n\n**In summary,** the image captures a person likely working with the Google AI Studio interface, potentially engaged in a coding or prompt engineering task, while simultaneously participating in a video call with Anand Subramanian.](https://youtu.be_2G1LqS6qO5s)

Key benefits:

- No setup cost or authentication handling
- Works with any visible screen content
- Full control over data exposure
- Extremely cost-effective (< $0.001 per short video)
- Bypasses anti-scraping measures
- Handles varying formats and layouts

### Quick Start Example

Here's a basic workflow using Google's AI Studio and Gemini:

1. **Record the Screen**
   - Use QuickTime (Mac) or Windows Game Bar (Windows), Screen2Gif, or any tool of your choice
   - Select specific screen area containing target data
   - Record scrolling/clicking through content
   - Keep recordings short (30-60 seconds)
2. **Process with Gemini**
   - Upload to [Google AI Studio](https://makersuite.google.com/app/prompts)
   - Select Gemini 1.5 Flash (cost-effective)
   - Prompt for structured output (JSON/CSV)

Example prompt for extracting tabular data:

```text
Turn this video into a JSON array where each item has:
{
  "date": "yyyy-mm-dd",
  "amount": float
}
```

### Cost Calculation

Gemini 1.5 Flash pricing (as of January 2025):

- $0.075 per million tokens
- Cost per frame ~ 250 tokens
- Cost for 24 hours of video at 1 frame per second ~ $1.62!

### Best Practices

1. **Recording Quality**
   - Frame only relevant content
   - Pause briefly on important data
   - Maintain consistent scroll speed
   - Use high contrast display settings
2. **Data Validation**
   - Always verify critical data manually
   - Use spot-checking for large datasets
   - Consider running multiple passes
   - Log and review any anomalies
3. **Error Handling**
   - Request data in simple formats (CSV/JSON)
   - Include validation in prompts
   - Split long videos into segments
   - Handle missing/partial data gracefully

### Use Cases

1. **Data Extraction**
   - Email content aggregation
   - Dashboard metrics collection
   - Protected web content
   - Legacy system data
2. **Data Journalism**
   - Public records analysis
   - Time-series data collection
   - Interactive visualization data
   - Government website scraping
3. **Business Intelligence**
   - Competitor pricing analysis
   - Market research data
   - Internal system migration
   - Legacy report conversion

Tools:

- [Google AI Studio](https://aistudio.google.com/app/prompts): Process videos with Gemini
- [QuickTime Player](https://support.apple.com/guide/quicktime-player/welcome/mac): Screen recording (Mac)
- [Screen2Gif](https://www.screentogif.com/): Screen recording (Windows)
- [OBS Studio](https://obsproject.com/): Advanced screen recording (cross-platform)

References:

- [Simon Willison's Video Scraping Tutorial](https://simonwillison.net/2024/Oct/17/video-scraping/)
- [Gemini API Documentation](https://ai.google.dev/docs)
