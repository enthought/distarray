class RandomModule(object):

    def __init__(self, context):
        self.context = context
        self.context._execute('import distarray.random')

    def rand(self, size=None, dist={0:'b'}, grid_shape=None):
        keys = self.context._key_and_push(size, dist, grid_shape)
        new_key = self.context._generate_key()
        subs = (new_key,) + keys + (self.context._comm_key,)
        self.context._execute(
            '%s = distarray.random.rand(%s,%s,%s,%s)' % subs
        )
        return DistArray(new_key, self.context)

    def normal(self, loc=0.0, scale=1.0, size=None, dist={0:'b'}, grid_shape=None):
        keys = self.context._key_and_push(loc, scale, size, dist, grid_shape)
        new_key = self.context._generate_key()
        subs = (new_key,) + keys + (self.context._comm_key,)
        self.context._execute(
            '%s = distarray.random.normal(%s,%s,%s,%s,%s,%s)' % subs
        )
        return DistArray(new_key, self.context)

    def randint(self, low, high=None, size=None, dist={0:'b'}, grid_shape=None):
        keys = self.context._key_and_push(low, high, size, dist, grid_shape)
        new_key = self.context._generate_key()
        subs = (new_key,) + keys + (self.context._comm_key,)
        self.context._execute(
            '%s = distarray.random.randint(%s,%s,%s,%s,%s,%s)' % subs
        )
        return DistArray(new_key, self.context)

    def randn(self, size=None, dist={0:'b'}, grid_shape=None):
        keys = self.context._key_and_push(size, dist, grid_shape)
        new_key = self.context._generate_key()
        subs = (new_key,) + keys + (self.context._comm_key,)
        self.context._execute(
            '%s = distarray.random.randn(%s,%s,%s,%s)' % subs
        )
        return DistArray(new_key, self.context)
