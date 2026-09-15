# -*- coding: UTF-8 -*-
from agents.base import SequentialFineTune
from agents.er import ExperienceReplay
from agents.ewc import EWC
from agents.lwf import LwF
from agents.mas import MAS
from agents.si import SI
from agents.dt2w import DT2W
from agents.aser import ASER
from agents.herding import Herding
from agents.inversion import Inversion
from agents.clops import CLOPS
from agents.der import DarkExperienceReplay
from agents.ders import DERS
from agents.gr import GenerativeReplay
from agents.er_sub import ER_on_Subject
from agents.fast_icarl import FastICARL
from agents.adr import ADR
from agents.pass_agent import PASS
from agents.il2a_agent import IL2A


agents = {'SFT': SequentialFineTune,
          'ER': ExperienceReplay,
          'ER_Sub': ER_on_Subject,
          'EWC': EWC,
          'LwF': LwF,
          'SI': SI,
          'MAS': MAS,
          'DT2W': DT2W,
          'ASER': ASER,
          'Herding': Herding,
          'Mnemonics': Herding,
          'Inversion': Inversion,
          'CLOPS': CLOPS,
          'DER': DarkExperienceReplay,
          'DERS': DERS,
          'GR': GenerativeReplay,
          'FastICARL': FastICARL,
          'ADR': ADR,
          'PASS': PASS,
          'IL2A': IL2A
          }

agents_replay = ['ER', 'Herding', 'Mnemonics', 'ASER', 'Inversion', 'CLOPS', 'GR', 'ER_Sub', 'FastICARL', 'DER', 'DERS']
agents_analytic = ['ADR']  # Analytic learning methods (no gradient descent in incremental phase)
