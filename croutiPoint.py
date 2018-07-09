import trueskill

class CroutiPoint:

    def __init__(self, name, pixels):

        self.rating = trueskill.Rating()
        self.name   = name
        self.pixels = pixels

