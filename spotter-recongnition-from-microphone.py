#!/usr/bin/env python3

# Real-time speech recognition from a microphone with sherpa-onnx Python API
# with endpoint detection.
#
# Note: This script uses ALSA and works only on Linux systems, especially
# for embedding Linux systems and for running Linux on Windows using WSL.
#
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
# to download pre-trained models

import argparse
import sys
from pathlib import Path
import sherpa_onnx

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)
import websocket
import time
import threading

get_result_event = threading.Event()

def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def messageSender():
    global recording_results
    websocket.enableTrace(False)
    server = "ws://192.168.1.124:8000/voicechat"
    ws = websocket.WebSocketApp(server)

    def on_open(ws):
        print("\n[INFO]messageSender: Websocket on open")
        while not get_result_event.is_set():
            # print("waiting for voice message...")
            time.sleep(1)
        print("\n[INFO]messageSender: Send %s to %s", recording_results, server)
        ws.send(recording_results)
        get_result_event.clear()

    def on_message(ws, message):
        print("\n[INFO]messageSender: Received from server:", message)
        on_open(ws)
        
    def on_close(ws,close_code, close_reason):
        print("\n[INFO]messageSender: WebSocket connection closed")
        on_open(ws)


    ws.on_open = on_open
    ws.on_message = on_message
    ws.on_close = on_close

    ws.run_forever()

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt",
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default="./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx",
        help="Path to the encoder model",
    )

    parser.add_argument(
        "--decoder",
        type=str,
        default="./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx",
        help="Path to the decoder model",
    )

    parser.add_argument(
        "--joiner",
        type=str,
        default="./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx",
        help="Path to the joiner model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values are greedy_search and modified_beam_search",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--hotwords-file",
        type=str,
        default="",
        help="""
        The file containing hotwords, one words/phrases per line, and for each
        phrase the bpe/cjkchar are separated by a space. For example:

        ▁HE LL O ▁WORLD
        你 好 世 界
        """,
    )

    parser.add_argument(
        "--hotwords-score",
        type=float,
        default=1.5,
        help="""
        The hotword score of each token for biasing word/phrase. Used only if
        --hotwords-file is given.
        """,
    )

    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="""
        The penalty applied on blank symbol during decoding.
        Note: It is a positive value that would be applied to logits like
        this `logits[:, 0] -= blank_penalty` (suppose logits.shape is
        [batch_size, vocab] and blank id is 0).
        """,
    )

    parser.add_argument(
        "--device-name",
        type=str,
        help="""
The device name specifies which microphone to use in case there are several
on your system. You can use

  arecord -l

to find all available microphones on your computer. For instance, if it outputs

**** List of CAPTURE Hardware Devices ****
card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

and if you want to select card 3 and device 0 on that card, please use:

  plughw:3,0

as the device_name.
        """,
    )

    return parser.parse_args()


def create_keyword_spotter():
    tokens              = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt"
    encoder             = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx"
    decoder             = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx"
    joiner              = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx"
    num_threads         = 1
    max_active_paths    = 4
    keywords_file       = "keywords.txt"
    keywords_score      = 1.0
    keywords_threshold  = 0.25
    num_trailing_blanks = 1
    provider            = "cpu"

    assert_file_exists(tokens)
    assert_file_exists(encoder)
    assert_file_exists(decoder)
    assert_file_exists(joiner)
    assert Path(
        keywords_file
    ).is_file(), (
        f"keywords_file : {keywords_file} not exist, please provide a valid path."
    )

    keyword_spotter = sherpa_onnx.KeywordSpotter(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        num_threads=num_threads,
        max_active_paths=max_active_paths,
        keywords_file=keywords_file,
        keywords_score=keywords_score,
        keywords_threshold=keywords_threshold,
        num_trailing_blanks=num_trailing_blanks,
        provider=provider,
    )

    return keyword_spotter

def create_recognizer(args):
    assert_file_exists(args.encoder)
    assert_file_exists(args.decoder)
    assert_file_exists(args.joiner)
    assert_file_exists(args.tokens)
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,  # it essentially disables this rule
        decoding_method=args.decoding_method,
        provider=args.provider,
        hotwords_file=args.hotwords_file,
        hotwords_score=args.hotwords_score,
        blank_penalty=args.blank_penalty,
    )
    return recognizer


def main():
    global recording_results
    args = get_args()
    # device_name = args.device_name
    # print(f"device_name: {device_name}")
    # alsa = sherpa_onnx.Alsa(device_name)
    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')


    print("Creating keyword_spotter")
    keyword_spotter = create_keyword_spotter()
    print("Creating recognizer")
    recognizer = create_recognizer(args)
    print("Started! Please speak")

    sample_rate = 48000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms

    spotter_stream = keyword_spotter.create_stream()
    stream = recognizer.create_stream()

    last_result = ""
    segment_id = 0
    wait_for_keyword = True
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            # samples = alsa.read(samples_per_read)  # a blocking read
            samples, _ = s.read(samples_per_read)
            samples = samples.reshape(-1)
            
            if wait_for_keyword:
                # print("wait for keyword...")
                spotter_stream.accept_waveform(sample_rate, samples)
                while keyword_spotter.is_ready(spotter_stream):
                    keyword_spotter.decode_stream(spotter_stream)
                result = keyword_spotter.get_result(spotter_stream)
                if result:
                    wait_for_keyword = False
                    print("\r{}".format(result), flush=True)
            else :
                # print("spott! please speak...")
                stream.accept_waveform(sample_rate, samples)
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                is_endpoint = recognizer.is_endpoint(stream)

                result = recognizer.get_result(stream)

                if result and (last_result != result):
                    last_result = result
                    print("\r{}:{}".format(segment_id, result), end="", flush=True)
                if is_endpoint:
                    if result:
                        print("\r{}:{}".format(segment_id, result), flush=True)
                        segment_id += 1
                        wait_for_keyword = True
                        recording_results = result
                        get_result_event.set()
                    recognizer.reset(stream)


if __name__ == "__main__":
    try:
        sender = threading.Thread(target=messageSender, name='messageSender')
        sender.start()
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
