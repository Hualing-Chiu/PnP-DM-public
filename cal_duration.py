import os
import torchaudio

def count_audio_and_duration(folder_path, target_sr=16000, extensions=('.wav', '.flac')):
    total_duration = 0.0
    file_count = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(root, file)
                try:
                    info = torchaudio.info(file_path)
                    duration = info.num_frames / info.sample_rate
                    total_duration += duration
                    file_count += 1
                except Exception as e:
                    print(f"Skipped {file_path}: {e}")

    print(f"🎧 Audio files: {file_count}")
    print(f"⏱️  Total duration: {total_duration / 3600:.2f} hours ({total_duration:.2f} seconds)")

# 範例使用：
count_audio_and_duration("/home/hualing/PnP-DM-public/results_vctk_new_model_2spk_all/source_separation/original/")