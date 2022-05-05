Our development set have 16 talks for you to evaluate your system.

# Two tasks
We have 2 tasks for Zh->En simultaneous translation:
1. Zh->En Translation, input: streaming transcription
2. Zh->En Translation, input: audio file

For each task, you can choose to input transcription(streaming_transcription/) or audio, depends on the track you participated in.

## Format description
### Streaming Transcription
Streaming transcription provides the golden transcripts in streaming format, where each sentence is broken into lines whose length is incremented by one word until the end of the sentence.


# How to evaluate BLEU
We provide reference_eval/ for you to evaluate the translation quality of your system.