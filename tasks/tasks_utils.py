# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

# summarization
from tasks.cnn_dailymail import TaskCnnDailymail
from tasks.xsum import TaskXSum
from tasks.xlsum import TaskXlSum
from tasks.dialogsum import TaskDialogSum
from tasks.wiki_lingua import TaskWikiLingua

# qa
from tasks.bbh import TaskBbh
from tasks.mmlu import TaskMmlu
from tasks.mmlu_pro import TaskMmluPro

# math
from tasks.gsm8k import TaskGSM8K
from tasks.gsm8k_platinum import TaskGSM8KPlatinum
from tasks.math500 import TaskMATH500


SUM_CLASS_DICT = {
    # summarization
    "cnn_dailymail": TaskCnnDailymail,
    "xsum": TaskXSum,
    "xlsum": TaskXlSum,
    "dialogsum": TaskDialogSum,
    "wiki_lingua": TaskWikiLingua,
}

QA_CLASS_DICT = {
    # qa
    "bbh": TaskBbh,
    "mmlu": TaskMmlu,
    "mmlu_pro": TaskMmluPro,
}

MATH_CLASS_DICT = {
    # math
    "gsm8k": TaskGSM8K,
    "gsm8k_platinum": TaskGSM8KPlatinum,
    "math500": TaskMATH500,
}

TASK_CLASS_DICT = {
    **SUM_CLASS_DICT,
    **QA_CLASS_DICT,
    **MATH_CLASS_DICT,
}

TASK_TYPE_DICT = {
    # summarization
    "cnn_dailymail": "sum",
    "xsum": "sum",
    "xlsum": "sum",
    "dialogsum": "sum",
    "wiki_lingua": "sum",

    # qa
    "bbh": "qa",
    "mmlu": "qa",
    "mmlu_pro": "qa",

    # math
    "gsm8k": "math",
    "gsm8k_platinum": "math",
    "math500": "math",
}
