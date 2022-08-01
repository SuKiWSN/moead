class Test:
    def __init__(self):
        self.a = 100
        self.b = 100
    
    def add(self):
        return self.a +self.b

if __name__ == "__main__":
    a = Test()
    print(a.add())