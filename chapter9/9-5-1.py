from textblob import TextBlob

texts = [
    "The service was very unfriendly. I will never use it again.",
    "서비스가 너무 불친절했어요. 다시는 이용 안 할 거예요.",
    "I had a really satisfying experience and I recommend it!",
    "정말 만족스러운 경험이었고 추천합니다!",
    "It was decent for the price.",
    "가격 대비 괜찮았어요.",
]

for text in texts:
    blob = TextBlob(text)
    decision = "positive" if blob.sentiment.polarity > 0 else "negative" if blob.sentiment.polarity < 0 else "neutral"
    print(f'\n문장: {text}')
    print(f' → 긍부정 판단: {decision} (감정 점수: {blob.sentiment.polarity * 100:.1f}%)')
    print(f" → 주관성 점수: {blob.sentiment.subjectivity*100:.1f}%\n")
