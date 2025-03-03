import re

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

def remove_punctuation(text):
    return re.sub(r'[^\w\s.!?‽？！，。]', '', text)


def trim_last_incomplete_confirmed_sentence(confirmed_transciption, confirm_offset_time):

    if len(confirmed_transciption) > 0:
        idx = len(confirmed_transciption) - 1
        while idx != -1:
            if confirm_offset_time == -1:
                if confirmed_transciption[idx].strip().endswith(('.', '?', '!')):
                    if idx != len(confirmed_transciption) - 1:
                        confirmed_transciption = confirmed_transciption[:idx + 1]
                    break

            elif confirmed_transciption[idx][1] == confirm_offset_time:
                if idx != len(confirmed_transciption) - 1:
                    confirmed_transciption = confirmed_transciption[:idx + 1]
                break
            idx -= 1

    return confirmed_transciption


def confirmation_process(non_confirmed_transcription, tokenize_transcription, confirmed_transciption, confirm_offset_time):
    
    sliced_tokenize_transcription = [(a,b, remove_punctuation(" ".join(t.split()))) for a,b,t in tokenize_transcription]

    confirmed_transciption = trim_last_incomplete_confirmed_sentence(confirmed_transciption, confirm_offset_time)

    if len(non_confirmed_transcription) == 0 or (len(sliced_tokenize_transcription) > 0 and non_confirmed_transcription[0][2] != sliced_tokenize_transcription[0][2]):
        non_confirmed_transcription = sliced_tokenize_transcription

    elif len(non_confirmed_transcription) > 0:
        print("11111111111111111111111111111")
        if confirm_offset_time == -1:
            if confirmed_transciption[-1][1] >= 5:
                confirm_offset_time = confirmed_transciption[-1][1]
        else:
            for ct_word in reversed(confirmed_transciption):
                if ct_word[1] - confirm_offset_time >= 5:
                    confirm_offset_time = ct_word[1]
                    break
        print("222222222222222222222222222")                    
        idx = 0
        for slice_idx in range(len(sliced_tokenize_transcription)):
            if sliced_tokenize_transcription[slice_idx][0] >= confirm_offset_time:
                idx = slice_idx
                break
        print("33333333333333333333333333")        
        while idx < min(len(non_confirmed_transcription), len(sliced_tokenize_transcription)):
            if (non_confirmed_transcription[idx][2]).lower() == (sliced_tokenize_transcription[idx][2]).lower():
                confirmed_transciption.append(non_confirmed_transcription[idx])
                idx += 1
            else:
                break
        print("44444444444444444444444444")        
        if idx > len(non_confirmed_transcription):
            non_confirmed_transcription.extend(sliced_tokenize_transcription[idx:])
        else:
            non_confirmed_transcription[idx:] = sliced_tokenize_transcription[idx:]

    return non_confirmed_transcription, confirmed_transciption, confirm_offset_time


def sentence_trim_buffer(tokenize_transcription, non_confirmed_transcription, confirmed_transcription, confirm_offset_time, sample_rate=SAMPLE_RATE, bytes_per_sample=BYTES_PER_SAMPLE):

    if len(confirmed_transcription) == 0:
        return 0, non_confirmed_transcription, confirm_offset_time  # No confirmed sentences to remove
    
    # Find the last confirmed sentence ending with ., ?, or !
    last_sentence = None
    for sentence in reversed(confirmed_transcription):
        if sentence.strip().endswith(('.', '?', '!')):
            last_sentence = sentence
            break

    if not last_sentence:
        return 0, non_confirmed_transcription, confirm_offset_time  # No complete sentence found, do not trim buffer
    
    confirmed_words = last_sentence.strip()
    end_time = None
    end_word_idx = -1

    # Find the corresponding end timestamp of the last word in the sentence
    for start, end, word in tokenize_transcription:
        if confirmed_words in remove_punctuation(word):
            end_time = end

    if end_time is None:
        return 0, non_confirmed_transcription, confirm_offset_time  # No match found, do not trim buffer
    
    for idx in range(len(non_confirmed_transcription)):
        if confirmed_words in non_confirmed_transcription[idx][2]:
            end_word_idx = idx

    if tokenize_transcription[-1][1] == end_time:
        return -1, non_confirmed_transcription[end_word_idx + 1:], confirm_offset_time
    
    # Compute bytes to remove
    bytes_to_remove = int(end_time * sample_rate * bytes_per_sample)

    # Trim buffer 
    return bytes_to_remove, non_confirmed_transcription[end_word_idx + 1:], -1


def threshold_trim_buffer(tokenize_transcription, non_confirmed_transcription, confirmed_transcription, buffer, sample_rate=SAMPLE_RATE, bytes_per_sample=BYTES_PER_SAMPLE):
    non_confirmed_transcription = []
    confirmed_transcription = trim_last_incomplete_confirmed_sentence(confirmed_transcription)
    end_time = 0

    while end_time < 30 and len(tokenize_transcription) > 0:
        a, end_time, word = tokenize_transcription.pop(0)
        confirmed_transcription.append(word.strip())
    
    confirmed_transcription[-1] = confirmed_transcription[-1] + ".."

    if len(tokenize_transcription) > 0:
        non_confirmed_transcription.extend([(a,b,remove_punctuation(" ".join(t.split()))) for a,b,t in tokenize_transcription])
        return buffer[int(end_time * sample_rate * bytes_per_sample):], confirmed_transcription, non_confirmed_transcription, -1

    return bytearray(), confirmed_transcription, non_confirmed_transcription, -1