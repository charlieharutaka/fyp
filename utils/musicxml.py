from collections import namedtuple
from dataclasses import dataclass
from typing import List
from xml.etree import ElementTree
from warnings import warn


from .arpabet import ARPABET, ARPABET_CONSONANTS


@dataclass
class Note:
    pitch: int
    duration: int
    lyric: List[str]


accidental2note = {'flat': -1, 'natural': 0, 'sharp': 1}
step2note = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def pitch2note(step, octave, accidental='natural'):
    try:
        step = str(step)
        octave = int(octave)
        accidental = str(accidental)
        return step2note[step] + accidental2note[accidental] + (octave + 1) * 12
    except KeyError:
        raise ValueError(f"Step {step} or accidental {accidental} is not valid")


def parse_musicxml(musicxml_file, constant_phoneme=None, lyric_number=2):
    root = ElementTree.parse(musicxml_file).getroot()
    tempo_elem = (root.findall(".//*[@tempo]"))
    if len(tempo_elem) > 1:
        warn("Only the first tempo element will be used", UserWarning)
    if len(tempo_elem) < 1:
        warn("Tempo will default to 80BPM", UserWarning)
        tempo = 80
    else:
        tempo = int(tempo_elem[0].attrib['tempo'])
    parts = {}
    for part in root.findall('part-list/score-part'):
        parts[part.attrib['id']] = {'name': part.find('part-name').text}
    for part_id in parts.keys():
        parts[part_id]['tempo'] = tempo
        part = root.find(f'part[@id="{part_id}"]')
        measures = part.findall('measure')
        parsed_notes = []
        last_lyric = []
        for measure in measures:
            notes = measure.findall('note')
            for note in notes:
                duration = int(note.find("duration").text)

                # We have to deal with tuplets
                time_mod = note.find("time-modification")
                if time_mod:
                    normal_notes = int(time_mod.find("normal-notes").text)
                    actual_notes = int(time_mod.find("actual-notes").text)
                    modification = normal_notes / actual_notes
                    duration *= modification

                is_rest = note.find("rest") is not None
                if not is_rest:
                    pitch = note.find("pitch")
                    step = pitch.find("step").text
                    octave = int(pitch.find("octave").text)
                    accidental = note.find("accidental")
                    if accidental is None:
                        accidental = 'natural'
                    else:
                        accidental = accidental.text
                    if constant_phoneme is None:
                        lyric = note.find("lyric[@number='2']")
                        if lyric is not None:
                            lyric = lyric.find("text").text.replace("\xa0", " ").split(" ")
                            for phoneme in lyric:
                                assert phoneme in ARPABET, f"Unexpected phoneme {phoneme}"
                        else:
                            assert len(last_lyric) > 0, "Expected a lyric"
                            lyric = []
                            last_phoneme = last_lyric[-1]
                            if last_phoneme in ARPABET_CONSONANTS:
                                # Pop it and carry it over, and repeat the vowel
                                lyric.append(last_lyric.pop(-1))
                                lyric.insert(0, last_lyric[-1])
                            else:
                                # Just repeat the vowel
                                lyric.append(last_lyric[-1])
                    else:
                        assert constant_phoneme in ARPABET, f"Unexpected phoneme {constant_phoneme}"
                        lyric = [constant_phoneme]
                    last_lyric = lyric
                    note = pitch2note(step, octave, accidental)
                    parsed_notes.append(Note(note, duration, lyric))
                else:
                    parsed_notes.append(Note(None, duration, None))
        parts[part_id]['notes'] = parsed_notes
    return parts