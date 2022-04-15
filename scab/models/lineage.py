#!/usr/bin/env python
# filename: lineage.py


#
# Copyright (c) 2021 Bryan Briney
# License: The MIT license (http://opensource.org/licenses/MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#




class Lineage():
    '''
    docstring for Lineage
    '''

    def __init__(self, adata):
        self.adata = adata.copy()
        self.obs = self.adata.obs


    def __iter__(self):
        for bcr in self.bcrs:
            yield bcr


    @property
    def bcrs(self):
        return self.adata.obs.bcr

    @property
    def pairs(self):
        return [b for b in self.bcrs if b.is_pair]

    @property
    def heavies(self):
        return [b.heavy for b in self.bcrs if b.heavy is not None]

    @property
    def lights(self):
        return [b.light for b in self.bcrs if b.light is not None]













