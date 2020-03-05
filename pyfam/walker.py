import numpy as np
import matplotlib.pyplot as plt

class walker:
    def __init__(self, x, y, model, p0, psig=None, cost=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.model = model

        #Define a cost function.
        if cost == None:
            self.cost = lambda p, x=x, y=y, m=model: np.sum(( m(x, p) - y)**2)
        else:
            self.cost = cost

        if type(psig)==type(None):
            self.psig=np.ones_like(p0)
        else:
            self.psig=np.array(psig).copy()

        self.move_to_p(p0)

        self.runs = {}

        self.accept_sample = [[] for a in range(len(self.p))]
        self.n_sample = 25

    def get_current_p(self):
        '''
        Return the current set of parameters that this walker sits at.
        '''
        return self.p

    def get_best_p(self, run_id, method="mean"):
        if method=="mean":
            return np.mean(self.runs[run_id], axis=0)
        if method=="recent":
            return self.runs[run_id][-1]

    def get_runs(self):
        return self.runs

    def move_to_p(self, p, p_cost=None):
        self.p = np.array(p).copy()
        if p_cost==None:
            self.c = self.cost(self.p)
        else:
            self.c = float(p_cost)

    def step(self, run_id=None):
        '''
        Pick a new prospective set of parameters and see how closely they fit the data.
        If it passes the acceptance criterion, move to these parameters.
        Otherwise, do nothing.
        '''
        p_order = np.random.choice(len(self.p), len(self.p), replace=False)
        cprosp_arr = []
        currc_arr = []
        lrat_arr = []
        lrat_cond_arr = []
        accrej_arr = []
        orig_p = self.p.copy()
        p_prosp = self.p.copy()

        #Step in all parameters individually one at a time.
        for i in p_order:
            p_prospective = self.p.copy()
            p_prospective[i] += np.random.normal(0, self.psig[i])
            p_prosp[i] = float(p_prospective[i])
            
            c_prosp = self.cost(p_prospective)
            likelihood_ratio = np.exp((-c_prosp + self.c))

            cond = np.random.uniform(0,1)
            if likelihood_ratio > cond:
                # New paramter was accepted
                self.move_to_p(p_prospective, c_prosp)
                self.accept_sample[i].append(1) # 1 for accepted steps
            else:
                # New paramter was rejected
                self.accept_sample[i].append(0) # 0 for rejected steps
                accrej_arr.append(0)
                self.p[i] = float(orig_p[i])

            # Update psig[i] value so that ~50% of steps are accepted.
            if len(self.accept_sample[i]) >= self.n_sample:
                if np.sum(self.accept_sample[i]) > 0:
                    self.psig[i] *= np.sum(self.accept_sample[i])/float(len(self.accept_sample[i])) / 0.5
                else:
                    self.psig[i] /= 2.0
                self.accept_sample[i] = []

        #Ensure that stored value for cost is correct.
        self.c = self.cost(self.p)

        if run_id!=None:
            if not run_id in self.runs:
                self.runs[run_id] = np.array( [np.insert(self.p, 0, self.c)] ) 
            else:
                self.runs[run_id] = np.vstack(( self.runs[run_id] , np.insert(self.p, 0, self.c) ))

    def walk(self, nsteps, run_id=None):
        if run_id==None:
            n=0
            while "walk"+str(n) in self.runs:
                n+=1
            run_id = "walk"+str(n)

        for i in range(nsteps):
            self.step(run_id)

    def get_mean(self):
        return np.mean(self.runs, axis=0)

    def plot_accepted(self, run_id, axes=None, **kwargs):
        params = self.runs[run_id][:,1:].T
        p_combs = list(combinations(params, 2))
        axis_labels = list(combinations(range(1,len(params)+1), 2))
        if type(axes) == type(None) or len(axes) != len(p_combs):
            axes = [plt.subplots()[1] for n in range(len(self.p))]

        for i in range(len(p_combs)):
            p1 = p_combs[i][0]
            p2 = p_combs[i][1]
            axes[i].set_xlabel('Q'+str(axis_labels[i][0]))
            axes[i].set_ylabel('Q'+str(axis_labels[i][1]))
            axes[i].scatter(p1, p2, **kwargs)
            axes[i].legend()
