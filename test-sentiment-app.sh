#!/bin/bash

POSITIVE_TEXTS=(
  "I absolutely love this product!"
  "Amazing experience, highly recommend."
  "Works like a charm, I'm very satisfied."
  "Exceptional quality and great value."
  "This exceeded all my expectations!"
)

NEGATIVE_TEXTS=(
  "I absolutely hate this product."
  "Terrible experience, not recommended."
  "Broke on the first day, waste of money."
  "Very disappointed with the quality."
  "It doesn't work as advertised."
)

NEUTRAL_TEXTS=(
  "It’s okay, nothing special."
  "Just average, works as expected."
  "Not bad, not great either."
  "Meh, I’ve seen better."
  "Does the job, no complaints."
)

NUM_POS=${1:-50}
NUM_NEG=${2:-50}
NUM_NEU=${3:-20}

URL="http://localhost:8080/api/v1/sentiment"

send_random_text() {
  local array_name=$1[@]
  local texts=("${!array_name}")
  local text="${texts[RANDOM % ${#texts[@]}]}"
  curl -s -X POST $URL \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"$text\"}" >> sentiment_log.txt
  echo "" >> sentiment_log.txt
  sleep 0.5
}

for ((i=0; i<NUM_POS; i++)); do
  send_random_text POSITIVE_TEXTS
done

for ((i=0; i<NUM_NEG; i++)); do
  send_random_text NEGATIVE_TEXTS
done

for ((i=0; i<NUM_NEU; i++)); do
  send_random_text NEUTRAL_TEXTS
done

curl -s -X POST "$URL/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts":[
    "I love this product! It works perfectly.",
    "This is terrible, it broke after one day.",
    "It is okay, nothing special but gets the job done.",
    "Best purchase ever! Highly recommended!",
    "Disappointed with the quality, not worth the money."
  ]}' >> sentiment_log.txt
echo "" >> sentiment_log.txt

echo "Simulation completed. Check sentiment_log.txt for results"
