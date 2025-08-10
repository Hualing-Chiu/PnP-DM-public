import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os

# === 步驟 1: 載入音檔 ===
def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

# === 步驟 2: 加入高斯噪音 ===
def add_gaussian_noise(audio, snr_db):
    rms = np.sqrt(np.mean(audio ** 2))
    snr_linear = 10 ** (snr_db / 10)
    noise_std = rms / np.sqrt(snr_linear)
    noise = np.random.normal(0, noise_std, audio.shape)
    return audio + noise

# === 步驟 3: 畫出頻譜圖 ===
def plot_spectrogram(audio, sr, save_path, title="Spectrogram"):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# === 步驟 4: 逐步加入噪音 + 存音檔 + 畫圖 ===
def progressive_noise_addition(audio, sr, output_dir, steps=4, snr_range=(20, 0)):
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    spec_dir = os.path.join(output_dir, "spectrogram")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)

    sf.write(os.path.join(audio_dir, "original.wav"), audio, samplerate=sr)
    plot_spectrogram(audio, sr, os.path.join(spec_dir, "original_spectrogram.png"), title="Original Audio Spectrogram")
    snr_levels = np.linspace(snr_range[0], snr_range[1], steps)

    for i, snr in enumerate(snr_levels):
        noisy_audio = add_gaussian_noise(audio, snr_db=snr)
        audio_path = os.path.join(audio_dir, f"noisy_step_{i+1}_snr{int(snr)}.wav")
        spec_path = os.path.join(spec_dir, f"spectrogram_step_{i+1}_snr{int(snr)}.png")

        sf.write(audio_path, noisy_audio, samplerate=sr)
        plot_spectrogram(noisy_audio, sr, spec_path, title=f"Step {i+1} - SNR {int(snr)} dB")

        print(f"Saved audio: {audio_path}")
        print(f"Saved spectrogram: {spec_path}")

# === 步驟 5: 執行流程 ===
if __name__ == "__main__":
    input_path = "/home/hualing/PnP-DM-public/results_libritts_new_model_2spk_all/source_separation/original/Sample_947_2.wav"  # 替換為你的音檔路徑
    output_folder = "/home/hualing/PnP-DM-public/noisy_outputs_1"
    sr = 16000

    clean_audio = load_audio(input_path, sr=sr)
    progressive_noise_addition(clean_audio, sr, output_dir=output_folder)