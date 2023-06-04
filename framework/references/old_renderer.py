import textwrap as tw
import matplotlib.pyplot as mpl
import matplotlib.ticker as ticker
import csv
import numpy as np
import itertools as it
import time
import operator as op
from . import config


class Outputter:
    """
    Handles opening and closing of the output file, as well as writing
    data in CSV format to the output file.

    :param file_path: Path to the file to be opened.
    :type file_path: string

    :param extra:
    """

    def __init__(self, file_path, extra=()):

        self.file_path = file_path
        self.file = open(file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.file)
        self.extra = extra

    def render(self, sample, time, total, uncertainty, extra=()):
        """
        Writes a line of data, typically for a single measurement, in
        CSV format, to the output file.

        :param total: The totals (or estimated mean total) from each
            of the 8 channels..
        :type total: sequence of length 8

        :param uncertainty: The uncertainties (or standard errors)
            from each of the 8 channels.
        :type uncertainty: sequence of length 8
        """
        sample = sample[-1]
        time = time[-1]
        total = total[-1]
        uncertainty = uncertainty[-1]
        extra = map(op.itemgetter(-1), extra)

        values = np.column_stack((total, uncertainty)).flatten()
        row = it.chain((sample, time), values, extra)

        self.csv_writer.writerow(row)
        self.file.flush()

    def summary(self, total, uncertainty, extra=()):
        values = np.column_stack((total, uncertainty)).flatten()
        extra = map(op.itemgetter(-1), extra)
        row = it.chain(('summary', None), values, extra)
        self.csv_writer.writerow(row)

    def start(self):
        self.csv_writer.writerow(it.chain(('sample', 'time'), *zip(
            config.CHANNEL_KEYS,
            ('{} uncertainty'.format(key) for key in config.CHANNEL_KEYS)
        ), self.extra))
        self.file.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """
        Closes the output file handle and cleans up its business.
        """
        self.file.close()


class Plotter:
    """
    Handles live-update plotting.

    :param left_channels: A list of channels to plot on the left
        line-plot (counts against time).
    :type left_channels: sequence of integers

    :param right_channels: A list of channels to plot on the right
        line-plot (counts against time).
    :type right_channels: sequence of integers

    :param bar: Whether to enable the live bar plot of the counts.
    :type bar: boolean

    :param title: Title of the plot, if any.
    :type title: string

    :param plot_window: The maximum number of data points to display
        on the line plot at a time.  Once the number of data points
        collected exceeds this number, the line plot will only display
        the most recent ``plot_window``-many points, discarding
        earlier points.  When ``plot_window`` is set to 0, the limit
        is removed (i.e. the number of points plotted is unlimited).
    :type plot_window: integer

    :param refresh_rate: The delay interval (in seconds) per each
        redrawing of the animation/plotting frame.
    :type refresh_rate: number
    """

    COLORS = ('tab:red', 'tab:orange', 'tab:olive', 'tab:green',
              'tab:cyan', 'tab:blue', 'tab:purple', 'tab:pink')

    def __init__(self, left_channels, right_channels, bar,
                 title='', plot_window=0, refresh_rate=.01):

        self.left_channels = left_channels
        self.right_channels = right_channels
        self.bar = bar

        self.plot_window = plot_window
        self.refresh_rate = refresh_rate

        two_column_fake = self.left_channels and self.right_channels
        two_column = two_column_fake or self.bar
        two_row = (self.left_channels or self.right_channels) and self.bar

        self.rows = 2 if two_row else 1
        self.columns = 2 if two_column else 1
        fake_columns = 2 if two_column_fake else 1

        self.figure = mpl.figure(figsize=(6 * self.columns, 6 * self.rows))
        self.figure.suptitle(title)

        self.subplotters = []

        if self.left_channels:
            self.subplotters.append(
                self.LinePlotter(
                    left_channels,
                    self.figure.add_subplot(self.rows, fake_columns, 1),
                    plot_window=self.plot_window,
                    title='Channels {}'.format(', '.join(
                        config.CHANNEL_KEYS[channel]
                        for channel in self.left_channels
                    )),
                )
            )
        if self.right_channels:
            self.subplotters.append(
                self.LinePlotter(
                    right_channels,
                    self.figure.add_subplot(
                        self.rows, fake_columns, fake_columns),
                    plot_window=self.plot_window,
                    title='Channels {}'.format(', '.join(
                        config.CHANNEL_KEYS[channel]
                        for channel in self.right_channels
                    )),
                ),
            )
        if self.bar:
            self.subplotters.append(self.BarPlotter(
                tuple(range(4)),
                self.figure.add_subplot(
                    self.rows, self.columns, 3 if two_row else 1),
                title='Single counts',
            ))
            self.subplotters.append(self.BarPlotter(
                tuple(range(4, 8)),
                self.figure.add_subplot(
                    self.rows, self.columns, 4 if two_row else 2),
                title='Coincidence counts',
            ))

    def start(self):
        mpl.pause(self.refresh_rate)

    def render(self, sample, time, total, uncertainty):
        """

        """
        for subplotter in self.subplotters:
            subplotter.plot(sample, time, total, uncertainty)

        mpl.pause(self.refresh_rate)

    def save(self, image_path):
        self.figure.set_size_inches(8 * self.columns, 6 * self.rows)
        self.figure.savefig(image_path)

    def summary(self, mean, uncertainty):
        mpl.show()

    def close(self):
        pass

    class LinePlotter:

        def __init__(self, channels, axes, plot_window=0, title=''):

            self.channels = channels

            self.axes = axes
            self.plot_window = plot_window

            self.axes.set_title(title)
            self.axes.set_xlabel('sample #')
            self.axes.set_ylabel('counts')

            self.line = [None] * len(channels)

            for i, channel in enumerate(self.channels):
                self.line[i], = self.axes.plot(
                    (), (),
                    label=config.CHANNEL_KEYS[channel],
                    color=Plotter.COLORS[channel],
                )

            if self.channels:
                self.axes.legend(loc=1)

            self.axes.xaxis.set_major_locator(
                ticker.MaxNLocator(integer=True, min_n_ticks=1)
            )
            self.axes.yaxis.set_major_locator(
                ticker.MaxNLocator(integer=True, min_n_ticks=1)
            )

        def plot(self, sample, time, total, uncertainty):
            x = sample[-self.plot_window:]
            for channel, line in zip(self.channels, self.line):
                y = total[-self.plot_window:, channel]

                line.set_xdata(x)
                line.set_ydata(y)

            self.axes.relim()
            self.axes.autoscale_view()

    class BarPlotter:
        def __init__(self, channels, axes, title=''):
            self.channels = channels
            self.axes = axes

            self.axes.set_title(title)
            self.axes.set_xlabel('channel')
            self.axes.set_ylabel('counts')

            self.axes.yaxis.set_major_locator(
                ticker.MaxNLocator(integer=True, min_n_ticks=1)
            )

            self.bar = self.axes.bar(
                self.channels, np.zeros(len(self.channels)),
                yerr=np.zeros(len(self.channels)),
                tick_label=[config.CHANNEL_KEYS[channel]
                            for channel in self.channels],
                capsize=4,
                color=[Plotter.COLORS[channel] for channel in self.channels]
            )
            self.bar_line, self.bar_cap, (self.bar_errorbar,) = (
                self.bar.errorbar
            )
            self.bar_cap_lower, self.bar_cap_upper = self.bar_cap

            # self.bar_cap_lower.set_marker(None)
            # self.bar_cap_upper.set_marker(None)

        def plot(self, sample, time, total, uncertainty):
            sample = sample[-1]
            total = total[-1, self.channels]
            uncertainty = uncertainty[-1, self.channels]

            for channel, bar, y in zip(self.channels, self.bar, total):
                bar.set_height(y)

            self.bar_errorbar.set_segments(
                zip(
                    zip(self.channels, total-uncertainty),
                    zip(self.channels, total+uncertainty)
                )
            )

            self.bar_cap_lower.set_ydata(total-uncertainty)
            self.bar_cap_upper.set_ydata(total+uncertainty)

            self.axes.set_xticklabels(
                ['{}\n{:.0f}'.format(config.CHANNEL_KEYS[channel], t)
                 for channel, t, u in zip(self.channels, total, uncertainty)])

            self.axes.relim()
            self.axes.autoscale_view()


class Printer:

    def __init__(self, fill_width=70):
        self.fill_width = 70

    def fill_print(self, message):
        print(tw.fill(message, width=self.fill_width))

    def hline(self, character='-'):
        print(character * self.fill_width)

    def summary(self, mean, uncertainty):
        print('summary (mean ± uncertainty)')
        self.print_table(mean, uncertainty)

    def close(self):
        pass

    def start(self):
        pass

    def render(self, sample, time_, total, uncertainty):
        sample = sample[-1]
        time_ = time_[-1]
        total = total[-1]
        uncertainty = uncertainty[-1]

        print('sample #{} ({})' .format(sample, time.strftime(
            '%Y-%m-%d (%A), %I:%M:%S %p',
            time.localtime(time_)
        )))

        self.print_table(total, uncertainty)

    def print_table(self, total, uncertainty):

        print('-' * 70)
        for line in self.format_entry(total, uncertainty):
            print(line)
        print()

    @classmethod
    def format_entry(cls, total, uncertainty):
        max_total_len = max(map(len, ('{:.2f}'.format(t) for t in total)))
        for key, t, u in zip(config.CHANNEL_KEYS, total, uncertainty):
            yield (('{:<' + str(config.MAX_CHANNEL_KEY_LEN) + '} | '
                    '{:<' + str(max_total_len) + '.2f} ± {:.2f}')
                   .format(key, t, u))
