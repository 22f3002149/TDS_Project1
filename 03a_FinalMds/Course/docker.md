## Containers: Docker, Podman

[Docker](https://www.docker.com/) and [Podman](https://podman.io/) are containerization tools that package your application and its dependencies into a standardized unit for software development and deployment.

Docker is the industry standard. Podman is compatible with Docker and has better security (and a slightly more open license). In this course, we recommend Podman but Docker works in the same way.

Initialize the container engine:

```bash
podman machine init
podman machine start
```

Common Operations. (You can use `docker` instead of `podman` in the same way.)

```bash
# Pull an image
podman pull python:3.11-slim

# Run a container
podman run -it python:3.11-slim

# List containers
podman ps -a

# Stop container
podman stop container_id

# Scan image for vulnerabilities
podman scan myapp:latest

# Remove container
podman rm container_id

# Remove all stopped containers
podman container prune
```

You can create a `Dockerfile` to build a container image. Here's a sample `Dockerfile` that converts a Python script into a container image.

```dockerfile
FROM python:3.11-slim
# Set working directory
WORKDIR /app
# Typically, you would use `COPY . .` to copy files from the host machine,
# but here we're just using a simple script.
RUN echo 'print("Hello, world!")' > app.py
# Run the script
CMD ["python", "app.py"]
```

To build, run, and deploy the container, run these commands:

```bash
# Create an account on https://hub.docker.com/ and then login
podman login docker.io

# Build and run the container
podman build -t py-hello .
podman run -it py-hello

# Push the container to Docker Hub. Replace $DOCKER_HUB_USERNAME with your Docker Hub username.
podman push py-hello:latest docker.io/$DOCKER_HUB_USERNAME/py-hello

# Push adding a specific tag, e.g. dev
TAG=dev podman push py-hello docker.io/$DOCKER_HUB_USERNAME/py-hello:$TAG
```

Tools:

- [Dive](https://github.com/wagoodman/dive): Explore image layers
- [Skopeo](https://github.com/containers/skopeo): Work with container images
- [Trivy](https://github.com/aquasecurity/trivy): Security scanner

[Podman Tutorial Zero to Hero | Full 1 Hour Course: Here\'s a detailed description of the image:\n\n**Overall Impression:** The image appears to be a screenshot from a recorded presentation or tutorial. It features a speaker on stage with accompanying on-screen text.\n\n**Visual Elements:**\n\n* **Speaker:** A man with a beard and glasses is holding a microphone and speaking. He is wearing a black t-shirt and a lanyard with identification. He is standing on a stage in front of an audience (partially visible).\n* **On-Screen Text:** The left side of the image features text overlayed on a purple background. The text reads:\n * "Giuseppe Scaramuzzino" (speaker\'s name)\n * "Podman tutorial: from zero to hero" (title of the tutorial)\n * "Full Course in 1 Hour" (duration/description)\n * "amadeus" (possibly the event or organization hosting the tutorial)\n * A white logo featuring a stylized penguin/seal icon within a circular frame.\n* **Background:** The background behind the speaker is blurred, indicating an audience in a large room. \n* **Color Palette:** The image uses a color scheme of purple, white, and black.\n\n**Composition:** The speaker is positioned on the right side of the frame, while the text and logo occupy the left side, creating a balanced composition.\n\n**Inference:** Given the text, it’s likely that this image is from a technical tutorial focused on the Podman containerization tool. The speaker, Giuseppe Scaramuzzino, is presenting a comprehensive course on the subject.\n\n\n\n](https://youtu.be_YXfA5O5Mr18)

[Learn Docker in 7 Easy Steps - Full Beginner's Tutorial: Here\'s a detailed description of the image:\n\n**Overall Impression:** The image appears to be a thumbnail or promotional graphic for a tutorial or guide on how to Dockerize a Node.js application.\n\n**Key Elements:**\n\n* **Text:** \n * "HOW TO" is displayed in large, bold, yellow lettering at the top left.\n * "DOCKERIZE" is prominently featured in large, bold, white lettering within a black rectangle, positioned centrally.\n * "IN 7 EASY STEPS" is displayed below "DOCKERIZE", with "7" stylized as a number composed of stacked blocks.\n* **Logos/Icons:**\n * The Docker logo (a whale with containers on its back) is positioned to the left of the word "DOCKERIZE".\n * A Node.js logo (green with "node" written in white) is placed in the upper right corner, with a small logo of flame above.\n* **Color Scheme:** The dominant colors are a bright, vibrant blue and green. The use of yellow and white accents makes the text stand out.\n* **Background:** The background features a wave-like pattern and is split into two colors (blue and green).\n\n**Overall, the image is designed to be visually appealing and informative, clearly communicating the topic of the content.**](https://youtu.be_gAkwW2tuIqE)

- Optional: For Windows, see [WSL 2 with Docker getting started](https://youtu.be/5RQbdMn04Oc)
