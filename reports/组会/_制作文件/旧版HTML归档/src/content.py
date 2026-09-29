for part in ['content_01_02.py','content_03_04.py','content_05_06.py']:
    exec((OUT/'src'/part).read_text(),globals())
