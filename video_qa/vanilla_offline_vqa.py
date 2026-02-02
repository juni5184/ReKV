import numpy as np
import torch
from decord import VideoReader, cpu
from logzero import logger

from video_qa.base import BaseVQA, work

class VanillaVQA(BaseVQA):
    def video_close_qa(self, question, candidates, correct_choice):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    def load_video(self, video_path):
        # For Qwen2.5VL
        if hasattr(self.qa_model, "prepare_video_tensor"):
            return self.qa_model.prepare_video_tensor(
                video_path=video_path,
                fps=self.sample_fps,
            )
        # For other models
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= self.retrieve_size:
            frame_idx = np.arange(total_frames)
        else:
            frame_idx = torch.linspace(0, total_frames - 1, steps=self.retrieve_size).long()
        return vr.get_batch(frame_idx).asnumpy()

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # load and preprocess video frames for QA
        if 'video_path' in video_sample:
            video_path = video_sample['video_path']
            video_path = video_path.replace('data', '/scratch2/juni5184/datasets')
        else:
            video_path = f'/scratch2/jshyun/datasets/Video-MME/videos/{video_sample["videoID"]}.mp4'
        
        video = self.load_video(video_path)
        if video is None:
            logger.error(f"Video not found: {video_path}")
            return

        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
            video_tensor = video_tensor.permute(0, 3, 1, 2)
        else:
            video_tensor = video

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        self.qa_model.encode_video(video_tensor)

        # Process each question using the same video KV-cache
        if 'conversations' in video_sample:
            for sample in video_sample['conversations']:
                logger.debug(f'sample: {sample}')
                question = sample['question']
                answer = sample['answer']
                
                # QA
                if 'choices' in sample:  # CloseQA
                    choices = sample['choices']
                    if answer is None:  # FIXME: an ugly fix for some benchmarks do not provide GT
                        answer = choices[0]
                    correct_choice = self.choice_letters[choices.index(answer)]
                    qa_results = self.video_close_qa(question, choices, correct_choice)
                    self.record[(self.retrieve_size, self.chunk_size)].append({
                        'video_id': video_sample['video_id'],
                        'question': question,
                        'choices': choices,
                        'answer': answer,
                        'correct_choice': correct_choice,
                        'pred_answer': qa_results['pred_answer'],
                        'pred_choice': qa_results['pred_choice'],
                        'qa_acc': qa_results['acc'] * 100,
                    })

                if 'question_type' in sample:
                    self.record[(self.retrieve_size, self.chunk_size)][-1]['task'] = sample['question_type']
        else:
            # for videomme
            question = video_sample['question']
            answer = video_sample['answer']
            choices = video_sample['options']

            if isinstance(answer, str) and answer.strip() in self.choice_letters:
                correct_choice = answer.strip()
                answer_text = choices[self.choice_letters.index(correct_choice)].strip()

            qa_results = self.video_close_qa(question, choices, correct_choice)
            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample.get('videoID', video_sample.get('video_id', None)),
                'question': question,
                'choices': choices,
                'answer': answer_text,
                'correct_choice': correct_choice,
                'pred_answer': qa_results['pred_answer'],
                'pred_choice': qa_results['pred_choice'],
                'qa_acc': qa_results['acc'] * 100,
            }) 

if __name__ == "__main__":
    work(VanillaVQA)
