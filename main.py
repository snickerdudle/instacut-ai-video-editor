from video_processor import VideoProcessor, VideoProcessingConfig
from prompts import video_summarization_prompt

if __name__ == "__main__":
    vpc = VideoProcessingConfig(
        sampling_policy="e10",
        output_path=r"./processed/",
        prompt=video_summarization_prompt,
    )
    vp = VideoProcessor(vpc)

    videos = [
        "https://www.youtube.com/watch?v=M4xqSDkid1s",
        "https://www.youtube.com/watch?v=IoZri9hq7z4",
    ]

    summary = vp.summarize(videos)
    print(summary)
