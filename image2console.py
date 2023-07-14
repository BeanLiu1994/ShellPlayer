import cv2
from img2shell import clear, reset
from img2shell import Transformer as Base

def rgb(red, green, blue, string):
    return f'\x1b[48;2;{red};{green};{blue}m{string}'

class Transformer(Base):
    def color(self, img):
        img = cv2.resize(img,(self.strw,self.strh),interpolation=cv2.INTER_AREA)
        if img.shape[2] < 3:
            return self.gray(img)
        
        def calc_row(i):
            row = []
            for j in range(self.strw):
                row.append(rgb(img[i, j, 2], img[i, j, 1], img[i, j, 0], " "))
            return ''.join(row)

        string = [clear()]
        rows = self.executor.map(calc_row, range(self.strh))
        for row in rows:
            string.append(row)
        string.append(reset())
        return '\n'.join(string)