"""Demo: Synesthesia simulation — sound input generates color output."""

import torch
from brainnn.core.config import BrainConfig, ModalityType
from brainnn.synesthesia.network import SynestheticNet
from brainnn.generation.cross_modal_gen import CrossModalGenerator


def main():
    config = BrainConfig.default_three_modality()
    config.synesthesia.synesthesia_strength = 0.8  # Strong synesthesia

    net = SynestheticNet(config)
    generator = CrossModalGenerator(config)

    # Simulate: give an auditory input, see what visual output emerges
    auditory_input = torch.randn(1, 1, 16000)  # ~1 second of audio
    visual_input = torch.randn(1, 3, 64, 64)   # A visual scene

    inputs = {
        ModalityType.AUDITORY: auditory_input,
        ModalityType.VISUAL: visual_input,
    }

    with torch.no_grad():
        fused, per_modality = net(inputs)
        print(f"Fused latent shape: {fused.shape}")
        print(f"Per-modality outputs: {list(per_modality.keys())}")

        # Generate visual output from auditory-influenced fused representation
        visual_output = generator.generate(fused, ModalityType.VISUAL)
        print(f"Generated visual output shape: {visual_output.shape}")

        # Generate audio from the same fused representation
        audio_output = generator.generate(fused, ModalityType.AUDITORY)
        print(f"Generated audio output shape: {audio_output.shape}")

    print("\nSynesthesia demo complete!")
    print("The fused representation contains cross-modal information")
    print("thanks to lateral connections between encoder layers.")


if __name__ == "__main__":
    main()
