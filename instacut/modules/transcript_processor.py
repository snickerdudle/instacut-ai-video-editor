"""Transcript processor code"""
import re
from typing import List, Optional, Union


def secondsToMinutes(seconds: float) -> str:
    """Converts seconds representation (121.4) to minutes (2:01). If the time is
    greater than an hour, it will be represented as hours:minutes:seconds, with
    the hours being prepended to the minutes and the minutes and seconds both
    being 2 digits long.

    Args:
        seconds (float): Seconds representation of time.
    Returns:
        str: Minutes representation of time.
    """
    hours = int(seconds // 3600)
    minutes = int(seconds // 60) % 60
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02}:{seconds:02}"
    return f"{minutes}:{seconds:02}"


class TranscriptProcessor:
    def __init__(self, transcript: List[dict]):
        self.transcript = transcript
        self.interval_dict = self.convertDictToIntervals(transcript)

    def convertDictToIntervals(self, transcript: List[dict]) -> List[dict]:
        """Converts the list of dicts to a list of dicts with the start and end
        times converted to intervals.

        Args:
            transcript (List[dict]): The transcript to convert.

        Returns:
            dict[tuple]: The converted transcript.
        """
        output_dict = {}
        for i in transcript:
            start = i["start"]
            end = start + i["duration"]
            output_dict[(start, end)] = i["text"]
        return output_dict

    def cleanIntervalsText(self, intervals: List[dict]) -> List[dict]:
        """Cleans the text of the intervals.

        Args:
            intervals (List[dict]): The intervals to clean.

        Returns:
            List[dict]: The cleaned intervals.
        """
        for i in intervals:
            text = i["text"]
            text = text.replace("\n", " ")
            text = text.replace("\t", " ")
            text = text.replace("\xa0", " ")
            text = re.sub(r"\s{2,}", " ", text)
            i["text"] = text.strip()
        return intervals

    def joinIntervals(
        self, intervals: List[dict], text_delimiter: Optional[str] = " "
    ) -> dict:
        """Joins a list of intervals.

        Args:
            intervals (List[dict]): The intervals to join.

        Returns:
            dict: The joined interval.
        """
        output_dict = {}
        # First, clean the text.
        intervals = self.cleanIntervalsText(intervals)
        # Then, join the intervals.
        output_dict["start"] = intervals[0]["start"]
        output_dict["duration"] = (
            intervals[-1]["start"]
            + intervals[-1]["duration"]
            - output_dict["start"]
        )
        output_dict["text"] = text_delimiter.join(
            [i["text"] for i in intervals]
        )
        return output_dict

    def generateJustTextTranscript(self):
        """Generates a text transcript from the transcript, without any of the
        timestamps.

        Returns:
            str: The text transcript.
        """
        joined_interval = self.joinIntervals(self.transcript)
        return joined_interval["text"]

    def generateTimestampedTextTranscript(
        self, transcript: Optional[List[dict]] = None
    ):
        """Converts the transcript to a timestamped text transcript."""
        transcript = transcript or self.transcript
        output = ""
        for i in transcript:
            output += f"[{secondsToMinutes(i['start'])}] {i['text']}\n"
        return output.strip()

    def trimIntervals(
        self, n: Optional[int] = None, s: Optional[Union[int, float]] = None
    ):
        if s is not None and n is not None:
            raise ValueError("Only one of n and s can be specified.")

        if s is not None:
            return self.trimIntervalsToSLong(s)
        else:
            if n is None:
                return self.trimIntervalsToNJoined()
            else:
                return self.trimIntervalsToNJoined(n)

    def trimIntervalsToNJoined(self, n: Optional[int] = 2**5):
        """Joins the intervals so that there are only n intervals.

        This method takes an integer `n` as input and returns a list of
        dictionaries representing the joined intervals. The method first reads
        the last interval timestamp to get the total length of the transcript.
        It then calculates the length of each interval by dividing the total
        length by `n`. It creates a list of empty lists with length `n`. It then
        iterates through the intervals and adds them to the corresponding list
        in the intervals list. It does this by calculating the end time of each
        interval and finding the corresponding index in the intervals list. It
        then appends the interval to the corresponding list. Finally, it
        iterates through the intervals and joins them using the `joinIntervals`
        method and appends the joined interval to the output list. The output
        list is then returned.

        Args:
            n (int): The number of intervals to join to.
        Returns:
            List[dict]: The joined intervals.
        """
        # First, get the total length of the transcript by reading the longest
        # duration in the transcript.
        total_length = max(
            [i["start"] + i["duration"] for i in self.transcript]
        )
        # Then, get the length of each interval.
        interval_length = total_length / n
        # Then, create a list of the intervals.
        intervals = [[] for _ in range(n)]
        # Then, iterate through the intervals and add them to the corresponding
        # list in the intverals list.
        for i in self.transcript:
            interval_end_time = i["start"] + i["duration"]
            interval_index = min(
                int(interval_end_time // interval_length), n - 1
            )
            intervals[interval_index].append(i)
        # Then, iterate through the intervals and join them.
        output_intervals = []
        for interval_range in intervals:
            output_intervals.append(self.joinIntervals(interval_range))
        return output_intervals

    def trimIntervalsToSLong(self, s: Optional[Union[int, float]] = 10):
        """Joins the intervals so that each interval has a length of S.

        This method takes a float `s` as input and returns a list of dictionaries
        representing the joined intervals. The method first reads the last interval
        timestamp to get the total length of the transcript. It then creates a list
        of empty lists with length `n`. It then iterates through the intervals and
        adds them to the corresponding list in the intervals list. It does this by
        calculating the end time of each interval and finding the corresponding
        index in the intervals list. It then appends the interval to the
        corresponding list. Finally, it iterates through the intervals and joins
        them using the `joinIntervals` method and appends the joined interval to
        the output list. The output list is then returned.

        Args:
            S (float): The length of each resulting interval.

        Returns:
            List[dict]: The joined intervals.
        """
        # First, get the total length of the transcript by reading the longest
        # duration in the transcript.
        total_length = max(
            [i["start"] + i["duration"] for i in self.transcript]
        )
        # Then, calculate the number of intervals.
        n = int(total_length / s)
        # Then, create a list of the intervals.
        intervals = [[] for _ in range(n)]
        # Then, iterate through the intervals and add them to the corresponding
        # list in the intervals list.
        for i in self.transcript:
            interval_end_time = i["start"] + i["duration"]
            interval_index = min(int(interval_end_time // s), n - 1)
            intervals[interval_index].append(i)
        # Then, iterate through the intervals and join them.
        output_intervals = []
        for interval_range in intervals:
            output_intervals.append(self.joinIntervals(interval_range))
        return output_intervals
