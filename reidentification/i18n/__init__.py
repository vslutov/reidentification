#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""I18n package.
"""

import gettext
gettext.bindtextdomain('reidentification', '.')
gettext.textdomain('reidentification')
_ = gettext.gettext
