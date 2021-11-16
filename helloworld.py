from metaflow import step, FlowSpec

class HelloWorldFlow(FlowSpec):

    @step
    def start(self):
        """First step"""
        print("This is the start step")
        self.next(self.hello)

    @step
    def hello(self):
        """Just saying hi"""
        print("Hello World!")
        self.next(self.end)

    @step
    def end(self):
        """Finish line"""
        print("This is the end step")

if __name__ == '__main__':
    HelloWorldFlow()
