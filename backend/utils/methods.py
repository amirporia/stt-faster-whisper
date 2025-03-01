import re

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

def remove_punctuation(text):
    return re.sub(r'[^\w\s.!?‽？！，。]', '', text)


def trim_last_incomplete_confirmed_sentence(confirmed_transciption):

    if len(confirmed_transciption) > 0:
        idx = len(confirmed_transciption) - 1
        while idx != -1:
            if confirmed_transciption[idx].strip().endswith(('.', '?', '!')):
                if idx != len(confirmed_transciption) - 1:
                    confirmed_transciption = confirmed_transciption[:idx + 1]
                break
            idx -= 1
        if idx == -1:
            confirmed_transciption = [] 

    return confirmed_transciption


def confirmation_process(non_confirmed_transcription, tokenize_transcription, confirmed_transciption):

    sliced_tokenize_transcription = [remove_punctuation(" ".join(t.split())) for a,b,t in tokenize_transcription]

    confirmed_transciption = trim_last_incomplete_confirmed_sentence(confirmed_transciption)

    if len(non_confirmed_transcription) == 0 or (len(sliced_tokenize_transcription) > 0 and non_confirmed_transcription[0] != sliced_tokenize_transcription[0]):
        non_confirmed_transcription = sliced_tokenize_transcription

    elif len(non_confirmed_transcription) > 0:
        idx = 0
        while idx < min(len(non_confirmed_transcription), len(sliced_tokenize_transcription)):
            if (non_confirmed_transcription[idx]).lower() == (sliced_tokenize_transcription[idx]).lower():
                confirmed_transciption.append(non_confirmed_transcription[idx])
                idx += 1
            else:
                break
        if idx > len(non_confirmed_transcription):
            non_confirmed_transcription.extend(sliced_tokenize_transcription[idx:])
        else:
            non_confirmed_transcription[idx:] = sliced_tokenize_transcription[idx:]

    return non_confirmed_transcription, confirmed_transciption


def sentence_trim_buffer(tokenize_transcription, non_confirmed_transcription, confirmed_transcription, buffer, sample_rate=SAMPLE_RATE, bytes_per_sample=BYTES_PER_SAMPLE):

    if len(confirmed_transcription) == 0:
        return buffer, non_confirmed_transcription  # No confirmed sentences to remove
    
    # Find the last confirmed sentence ending with ., ?, or !
    last_sentence = None
    for sentence in reversed(confirmed_transcription):
        if sentence.strip().endswith(('.', '?', '!')):
            last_sentence = sentence
            break


    if not last_sentence:
        return buffer, non_confirmed_transcription  # No complete sentence found, do not trim buffer
    
    confirmed_words = last_sentence.strip()
    end_time = None
    end_word_idx = 0
    print(f"tokenize_transcription: {tokenize_transcription}")
    print(f"confirmed_words: {confirmed_words}")
    # Find the corresponding end timestamp of the last word in the sentence
    for start, end, word in tokenize_transcription:
        if confirmed_words in remove_punctuation(word):
            end_time = end
    print(f"end_time: {end_time}")
    for idx in range(len(non_confirmed_transcription)):
        if confirmed_words in non_confirmed_transcription[idx]:
            end_word_idx = idx
    
    if end_time is None:
        return buffer, non_confirmed_transcription  # No match found, do not trim buffer
    
    # Compute bytes to remove
    bytes_to_remove = int(end_time * sample_rate * bytes_per_sample)

    # Trim buffer  
    return buffer[bytes_to_remove:], non_confirmed_transcription[end_word_idx + 1:]


def threshold_trim_buffer(tokenize_transcription, non_confirmed_transcription, confirmed_transcription, buffer, sample_rate=SAMPLE_RATE, bytes_per_sample=BYTES_PER_SAMPLE):
    
    if int(len(buffer)/(sample_rate * bytes_per_sample)) >= 30 and len(tokenize_transcription) > 0:
        non_confirmed_transcription = []
        confirmed_transcription = trim_last_incomplete_confirmed_sentence(confirmed_transcription)
        end_time = 0
        while end_time < 30 and len(tokenize_transcription) > 0:
            a,end_time, word = tokenize_transcription.pop(0)
            confirmed_transcription.append(word.strip())

        if len(tokenize_transcription) > 0:
            non_confirmed_transcription.extend([remove_punctuation(" ".join(t.split())) for a,b,t in tokenize_transcription])

        bytes_to_remove = int(end_time * sample_rate * bytes_per_sample)
        
        return buffer[bytes_to_remove:], confirmed_transcription, non_confirmed_transcription
    
    return buffer, confirmed_transcription, non_confirmed_transcription


def model_transcribe(pcm_array, asr):

    transcription = asr.transcribe(pcm_array)
    tokenize_transcription = asr.ts_words(transcription)
    return tokenize_transcription