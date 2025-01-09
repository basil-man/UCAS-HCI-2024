import nltk
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np


def calculate_self_bleu(text):
    # Split text into sentences (Chinese specific)
    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if line:  # Skip empty lines
            # Split by common Chinese punctuation
            for sent in line.split("。"):
                if sent.strip():
                    sentences.append(sent.strip())

    if len(sentences) <= 1:
        return 0.0

    # Calculate BLEU scores for each sentence pair
    bleu_scores = []
    smoother = SmoothingFunction().method1  # Use smoothing to handle short sentences

    for i, reference in enumerate(sentences):
        reference_tokens = list(jieba.cut(reference))  # Use jieba for Chinese word segmentation

        for j, candidate in enumerate(sentences):
            if i != j:  # Don't compare sentence with itself
                candidate_tokens = list(jieba.cut(candidate))
                # Calculate BLEU score with smoothing
                score = sentence_bleu(
                    [reference_tokens],
                    candidate_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25),  # Use up to 4-grams
                    smoothing_function=smoother,
                )
                bleu_scores.append(score)

    # Calculate average Self-BLEU score
    avg_self_bleu = np.mean(bleu_scores)
    return avg_self_bleu


def analyze_text_diversity(text):
    self_bleu_score = calculate_self_bleu(text)
    diversity_score = 1 - self_bleu_score  # Convert to diversity score (higher is more diverse)

    print(f"Self-BLEU Score: {self_bleu_score:.4f}")
    print(f"Diversity Score: {diversity_score:.4f}")
    print("\nInterpretation:")
    if diversity_score >= 0.8:
        print("The text shows very high diversity in its content.")
    elif diversity_score >= 0.6:
        print("The text shows good diversity in its content.")
    elif diversity_score >= 0.4:
        print("The text shows moderate diversity in its content.")
    else:
        print("The text shows low diversity in its content.")


if __name__ == "__main__":
    # Example text (Chinese feedback on deadlift form)
    text = """用户在卧推时，小臂与大臂呈钝角，肩关节可能会处于一个相对不稳定的状态，增加肩关节的压力，且胸肌的激活会减少，三角肌和肱三头肌可能会代偿发力。请尽量保持90度发力。
用户在卧推时，膝盖弯曲角度过大，臀部会被迫处于过度屈曲状态。这样会导致臀部肌肉参与过多的发力，从而可能导致臀部和下背部的过度紧张或疲劳,同时下肢的肌肉群（尤其是大腿和臀部）会过度用力，这可能会打乱下肢和上肢之间的协调性。在卧推时，正确的膝盖弯曲应当有助于传递力到上肢，而过度弯曲可能会干扰力量传递的流畅性，影响卧推的推力效率。
用户在我卧推时，当杠铃处于最高点时，视线没有在杠铃正下方,视线的方向通常会影响头部和脖部的姿势。如果视线偏离杠铃，可能导致头部过度伸展或下压，这会影响颈部和脊柱的稳定性。长时间的不正确头部姿势可能引起脖部和背部的不适或疼痛。
用户在卧推时，双臂夹角过大会导致肩膀的外展角度过大，增加肩部关节的负担，尤其是肩关节前部的旋转袖肌群。这种姿势可能会导致肩部过度拉伸或受压，长期这样做容易引起肩部疼痛、炎症，甚至是肩袖撕裂等严重问题。如果双臂夹角过大，手肘过低，胸大肌的激活程度可能会减弱，因为胸肌在过度外展的情况下无法有效发挥作用。反而，三角肌和肩部的其他肌肉群会承担更多的压力，可能导致训练效果不理想，甚至造成肌肉不平衡。
用户在卧推时，手臂应当保持一条弧线上下升，避免直上直下，当你在卧推时，手肘会略微向外扩展，形成一个自然的弧线，这有助于保持肩关节的安全和稳定。直上直下的动作会让肩关节过度承受压力，增加受伤风险。如果手臂直上直下，肩膀的内旋角度会过大，肩部的前侧肌肉（如肩袖）会过度受力。手臂弯曲并沿着弧线升降时，可以让肩部承受的压力更加均匀，从而避免因过度压迫造成肩关节的不适或损伤。手臂沿着弧线升降有助于更好地激活胸大肌。在直上直下的动作中，胸肌的参与度较低，更多的压力会被转移到肩膀和三头肌上。而弧线运动能够有效地让胸大肌承受更多的负荷，增强训练效果。"""

    analyze_text_diversity(text)
