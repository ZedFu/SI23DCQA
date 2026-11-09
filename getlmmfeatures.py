"""Extract the original mPLUG-Owl2 hidden-state features for SI23DCQA."""

import argparse
from pathlib import Path

from data_utils import feature_dir, read_annotations

PROMPTS = {
    'video': 'How would you rate the quality of the object in this video?',
    'compare': 'What are the similarities in color, shape, and overall appearance of the objects shown in these two images?',
    'identity': 'Are the objects in these two images the same object?',
}
FILENAMES = {'video': '3D.pt', 'compare': 'compare2.pt', 'identity': 'compare.pt'}


def load_video(video_file):
    from decord import VideoReader
    from PIL import Image

    reader = VideoReader(str(video_file))
    fps = reader.get_avg_fps()
    if fps <= 0 or len(reader) == 0:
        raise ValueError(f'Invalid video: {video_file}')
    indices = [int(fps * i) for i in range(max(1, int(len(reader) / fps)))]
    return [Image.fromarray(frame) for frame in reader.get_batch(indices).asnumpy()]


def load_image_video(image_path, video_path):
    from decord import VideoReader
    from PIL import Image

    first_frame = VideoReader(str(video_path)).get_batch([0]).asnumpy()[0]
    return [Image.open(image_path).convert('RGB'), Image.fromarray(first_frame)]


def expand2square(image, background_color):
    from PIL import Image

    width, height = image.size
    side = max(width, height)
    result = Image.new(image.mode, (side, side), background_color)
    result.paste(image, ((side - width) // 2, (side - height) // 2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--ann', default='all-result.txt')
    parser.add_argument('--model-path', default='MAGAer13/mplug-owl2-llama2-7b')
    parser.add_argument('--model-base', default=None)
    parser.add_argument('--mode', choices=PROMPTS, default='compare')
    parser.add_argument('--device', default='cuda:0')
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument('--load-8bit', action='store_true')
    quantization.add_argument('--load-4bit', action='store_true')
    args = parser.parse_args()

    import torch
    from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from mplug_owl2.conversation import conv_templates
    from mplug_owl2.model.builder import load_pretrained_model
    from mplug_owl2.mm_utils import tokenizer_image_token

    # The loader checks for this architecture name; a Hugging Face repo name has hyphens.
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path, args.model_base, 'mplug_owl2',
        args.load_8bit, args.load_4bit, device=args.device,
    )
    model.eval()
    conversation = conv_templates['mplug_owl2'].copy()
    conversation.append_message(conversation.roles[0], PROMPTS[args.mode] + '\n' + DEFAULT_IMAGE_TOKEN)
    conversation.append_message(conversation.roles[1], None)
    input_ids = tokenizer_image_token(conversation.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors='pt').unsqueeze(0).to(args.device)
    root = Path(args.data_root)
    for record in read_annotations(root / args.ann):
        target = feature_dir(root, record['video_path'])
        target.mkdir(parents=True, exist_ok=True)
        if args.mode == 'video':
            images = load_video(root / record['video_path'])
        else:
            images = load_image_video(root / record['image_path'], root / record['video_path'])
        images = [expand2square(im, tuple(int(x * 255) for x in processor.image_mean)) for im in images]
        pixels = processor.preprocess(images, return_tensors='pt')['pixel_values'].half().to(args.device)
        with torch.inference_mode():
            outputs = model(input_ids, images=[pixels], output_hidden_states=True)
            features = outputs['hidden_states'][-1].mean(dim=1)
        if features.shape != (1, 4096):
            raise ValueError(f'Expected one 4096-dimensional feature, got {tuple(features.shape)}')
        torch.save(features.cpu(), target / FILENAMES[args.mode])
        print(target / FILENAMES[args.mode])


if __name__ == '__main__':
    main()
