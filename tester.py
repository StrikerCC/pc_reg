# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 8/19/21 5:22 PM
"""

from dataset import reader
from registration import registrations

statistics_reg = {global_registration: {'method': global_registration,
                                        '#case': 0,
                                        '#failure': 0,
                                        'time_global': 0.0,
                                        'time_local': 0.0,
                                        'error_t': 0.0,
                                        'error_o': 0.0
                                        } for global_registration in registrations}


class tester:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        # self.registration = None

    def start(self, regs):
        if not isinstance(regs, list):
            regs = [regs]
        for reg in regs:
            statistic = statistics_reg[reg]
            reg(self.dataloader, statistic, show_flag=True)
        self.__report(statistics_reg)

    def __report(self, statistic):
        pass


def main():
    sample_path = './data/TUW_TUW_models/TUW_models/'
    output_path = './data/TUW_TUW_data/'
    output_json_path = output_path + 'data.json'

    dl = reader()
    dl.read(output_json_path)
    reg_tester = tester(dataloader=dl)
    reg_tester.start(regs=registrations)


if __name__ == '__main__':
    main()
