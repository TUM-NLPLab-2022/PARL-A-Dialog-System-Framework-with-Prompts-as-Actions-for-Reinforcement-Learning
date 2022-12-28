"""
Code from https://github.com/florianvonunold/DYME
"""

import numpy as np


def cosine_similarity(a, b):
    return np.sum(a * b, axis=1) / np.sqrt((np.sum(a * a, axis=1) * np.sum(b * b, axis=1)))


def _get_emojis():
    # All emojis in the order returned by deepmoji
    EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: :pensive: " + \
             ":ok_hand: :blush: :heart: :smirk: :grin: :notes: :flushed: " + \
             ":100: :sleeping: :relieved: :relaxed: :raised_hands: " + \
             ":two_hearts: :expressionless: :sweat_smile: :pray: " + \
             ":confused: :kissing_heart: :heartbeat: :neutral_face: " + \
             ":information_desk_person: :disappointed: :see_no_evil: " + \
             ":tired_face: :v: :sunglasses: :rage: :thumbsup: :cry: " + \
             ":sleepy: :yum: :triumph: :hand: :mask: :clap: :eyes: :gun: " + \
             ":persevere: :smiling_imp: :sweat: :broken_heart: " + \
             ":yellow_heart: :musical_note: :speak_no_evil: :wink: :skull: " + \
             ":confounded: :smile: :stuck_out_tongue_winking_eye: :angry: " + \
             ":no_good: :muscle: :facepunch: :purple_heart: " + \
             ":sparkling_heart: :blue_heart: :grimacing: :sparkles:"
    EMOJIS = EMOJIS.split(' ')
    return EMOJIS


def _get_emoji_to_sentiment_dict():
    # How detected emojis map to sentiment score
    emoji_to_sentiment = {
        # very strongly positive
        ':kissing_heart:': 1, ':thumbsup:': 1, ':ok_hand:': 1,
        ':smile:': 1,

        # strongly positive
        ':blush:': 0.75, ':wink:': 0.75, ':muscle:': 0.75,
        ':grin:': 0.75, ':heart_eyes:': 0.75, ':100:': 0.75,

        # positive
        ':smirk:': 0.5, ':stuck_out_tongue_winking_eye:': 0.5,
        ':sunglasses:': 0.5, ':relieved:': 0.5, ':relaxed:': 0.5,
        ':blue_heart:': 0.5, ':two_hearts:': 0.5, ':heartbeat:': 0.5,
        ':yellow_heart:': 0.5,

        # negative
        ':disappointed:': -0.5, ':eyes:': -0.5,
        ':expressionless:': -0.5, ':sleeping:': -0.5,
        ':grimacing:': -0.5,

        # strongly negative
        ':neutral_face:': -0.75, ':confused:': -0.75,
        ':triumph:': -0.75, ':confounded:': -0.75,

        # very strongly negative
        ':unamused:': -1, ':angry:': -1,  # removing ':hand:': -1 due to ambiguity
        ':rage:': -1
    }
    return emoji_to_sentiment


def _get_sentiment_multiplier():
    EMOJIS = _get_emojis()
    emojis_to_sentiment = _get_emoji_to_sentiment_dict()
    sentiment_multiplier = np.zeros(len(EMOJIS))
    for emoji, reward_val in emojis_to_sentiment.items():
        loc = EMOJIS.index(emoji)
        sentiment_multiplier[loc] = reward_val
    return sentiment_multiplier
