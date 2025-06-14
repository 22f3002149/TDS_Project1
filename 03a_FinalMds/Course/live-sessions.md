# Live Sessions

Live sessions by the instructors and TAs are recorded and uploaded to YouTube.

[TDS Live Sessions: Jan 2025: Here\'s a detailed description of the image:\n\n**Overall Impression:** The image is a colorful, flat-design illustration representing the tools used in data science. It has a collage-like composition with various icons and graphics arranged on a muted orange background within a rectangular border.\n\n**Key Elements & Details:**\n\n* **Text:** Bold yellow text reads "TOOLS IN DATA SCIENCE" in the center of the illustration.\n* **Icons/Graphics:** A wide variety of icons are scattered throughout, symbolizing data science concepts:\n * **Data Visualization:** Bar graphs, line graphs, scatter plots.\n * **Machine Learning:** Neural network diagrams, algorithms.\n * **Big Data:** Network connections, data blocks.\n * **General Tools:** Gears, dials, pencils, monitors, and various abstract shapes.\n* **Color Palette:** The illustration employs a vibrant color scheme with blues, reds, yellows, and greens, giving it a modern and visually appealing look.\n* **Style:** The image is designed in a flat, geometric style, which is common in modern illustration.\n* **Composition:** The elements are arranged in a slightly chaotic but balanced way, creating a dynamic and engaging visual.\n\n**Overall, the image effectively conveys the idea of data science as a field that utilizes a wide array of tools and concepts.**](https://www.youtube.com_playlist?list=PL_h5u1jMeBCl1BquBhgunA4t08XAxsA-C)

These were downloaded using [yt-dlp](https://github.com/yt-dlp/yt-dlp). The options compress the audio optimized for speech.

```bash
yt-dlp --extract-audio --audio-format opus --embed-thumbnail --postprocessor-args \
  "-c:a libopus -b:a 12k -ac 1 -application voip -vbr off -ar 8000 -cutoff 4000 -frame_duration 60 -compression_level 10" \
  $YOUTUBE_URL
```

They were then transcribed by Gemini 1.5 Flash 002 (currently the best model from a price-quality perspective for audio transcription).

System prompt:

```
You are an expert transcriber of data science audio tutorials
```

User prompt:

```
Transcribe this audio tutorial about Tools in Data Science (TDS) as an FAQ.
Summarize the student questions faithfully.
Summarize the answers succinctly, without missing information, in a conversational style.
Avoid repeating questions, consolidating similar ones.
Prefer "You" and "I" instead of "student" and "instructor".
For example:

**Q1: [Concisely framed question]**

**A1:** [Succinct answer]
```
