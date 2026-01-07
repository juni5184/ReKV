import torch
from logzero import logger

from video_qa.base_vanilla import BaseVanillaVQA, work_vanilla


class VanillaVQA(BaseVanillaVQA):
    def video_close_qa(self, question, candidates, correct_choice):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # load and preprocess video frames for QA
        video_path = video_sample['video_path']
        video_path = video_path.replace('data', '/scratch2/juni5184/datasets')
        video = self.load_video(video_path)
        video_tensor = torch.from_numpy(video)

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
                self.record[self.chunk_size].append({
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
                self.record[self.chunk_size][-1]['task'] = sample['question_type']

if __name__ == "__main__":
    work_vanilla(VanillaVQA)
