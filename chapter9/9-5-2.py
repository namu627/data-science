from transformers import pipeline

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

nlp = pipeline(
    "text-classification",
    model=MODEL,
    tokenizer=MODEL,
    device=-1,           # CPU. GPU 쓰면 device_map="auto"
    return_all_scores=False
)
texts = [
    "The service was very unfriendly. I will never use it again.",
    "서비스가 너무 불친절했어요. 다시는 이용 안 할 거예요.",
    "I had a really satisfying experience and I recommend it!",
    "정말 만족스러운 경험이었고 추천합니다!",
    "It was decent for the price.",
    "가격 대비 괜찮았어요.",
]

for text in texts:
    out = nlp(text, truncation=True, max_length=128)[0]
    print(f"\n문장: {text}")
    print(f" → 긍부정 판단: {out['label']} (확률: {out['score'] * 100:.1f}%)\n")
