from typing import Optional, Tuple

import numpy as np
from typing_extensions import override

from audio_processing import Audio, FrameSeries

from .annotations import DatasetConstructor, MIDIAnnotations, MusicNetAnnotations


class MelSpectrumMuNDatasetConstructor(DatasetConstructor):
    @override
    def procedure(
        self,
        data_path: str,
        label_path: str,
        frame_length: int,
        frame_shift: int,
        bins: int,
        frames: int,
        fs: int = 16000,
        fft_point: Optional[int] = None,
        data_trim: Optional[Tuple[int, int]] = None,
        annotation_trim: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mel_spectrum = self.data_procedure(
            path=data_path,
            frame_length=frame_length,
            frame_shift=frame_shift,
            fft_point=fft_point if fft_point is not None else frame_length,
            bins=bins,
            trim=data_trim,
        )
        labels = self.annotation_procedure(
            path=label_path,
            frame_shift=frame_shift,
            fs=fs,
            num_frames=len(mel_spectrum),
            trim=annotation_trim,
        )

        return (
            mel_spectrum.to_patches(frames).squeeze(),
            labels.to_patches(frames).squeeze(),
        )

    @staticmethod
    def data_procedure(
        path: str,
        frame_length: int,
        frame_shift: int,
        fft_point: int,
        bins: int,
        trim: Optional[Tuple[int, int]] = None,
    ) -> FrameSeries:
        data = (
            Audio.read(path)
            .to_frames(frame_length, frame_shift)
            .to_spectrum(fft_point=fft_point)
            .to_amplitude()
            .to_mel(bins)
            .linear_to_dB()
        )

        data = data.trim_by_value(*trim) if trim is not None else data

        return data

    @staticmethod
    def annotation_procedure(
        path: str,
        frame_shift: int,
        fs: int,
        num_frames: int,
        trim: Optional[Tuple[int, int]] = None,
    ) -> FrameSeries:
        annotations = MusicNetAnnotations.from_csv(path, fs).to_frames(
            fs=fs, num_frames=num_frames, frame_shift=frame_shift
        )
        annotations = (
            FrameSeries(annotations).trim_by_value(*trim)
            if trim is not None
            else FrameSeries(annotations)
        )

        return annotations


class CQTMuNDatasetConstructor(DatasetConstructor):
    @override
    def procedure(
        self,
        data_path: str,
        annotation_path: str,
        frame_shift: int,
        fmin: float,
        octaves: int,
        bins_per_octave: int,
        frames: int,
        power: bool = False,
        fs: int = 16000,
        data_trim: Optional[Tuple[int, int]] = None,
        annotation_trim: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = self.data_procedure(
            path=data_path,
            frame_shift=frame_shift,
            fmin=fmin,
            octaves=octaves,
            bins_per_octave=bins_per_octave,
            power=power,
            trim=data_trim,
        )

        annotations = self.annotation_procedure(
            path=annotation_path,
            frame_shift=frame_shift,
            fs=fs,
            num_frames=data.shape[0],
            trim=annotation_trim,
        )

        return (
            data.to_patches(frames).squeeze(),
            annotations.to_patches(frames).squeeze(),
        )

    @staticmethod
    def data_procedure(
        path: str,
        frame_shift: int,
        fmin: float,
        octaves: int,
        bins_per_octave: int,
        power: bool,
        trim: Optional[Tuple[int, int]] = None,
    ) -> FrameSeries:
        data = (
            Audio.read(path)
            .to_cqt(
                frame_shift=frame_shift,
                fmin=fmin,
                n_bins=octaves * bins_per_octave,
                bins_per_octave=bins_per_octave,
            )
            .to_amplitude()
        )
        data = data.linear_to_power() if power else data
        data = data.trim_by_value(*trim) if trim is not None else data

        return data

    @staticmethod
    def annotation_procedure(
        path: str,
        frame_shift: int,
        fs: int,
        num_frames: int,
        trim: Optional[Tuple[int, int]] = None,
    ) -> FrameSeries:
        annotations = MusicNetAnnotations.from_csv(path, fs).to_frames(
            fs=fs, num_frames=num_frames, frame_shift=frame_shift
        )
        annotations = (
            FrameSeries(annotations).trim_by_value(*trim)
            if trim is not None
            else FrameSeries(annotations)
        )

        return annotations
