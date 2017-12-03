import numpy as np
from itertools import combinations
from scipy.misc import comb
import matplotlib.pyplot as plt
plt.ion()
import glob

from walker import walker
from contour import contour

class MCMC:
    def __init__(self, x, y, model, savepath=None, cost=None, pnames=None):
        '''
        x, y      - Data to be fit.
                    type(x) and type(x) should be an array, list, or other array-like object.
        
        model     - Function that takes a set of the x-data and a set of parameters
                    and returns a model of the y data.
                        ex. y_guess = model(x, p_guess), for some set of parameters, p_guess.
                    model should be a function, class method, or other callable object.

        savepath  -  

        cost      - Optional. Function used to determine goodness-of-fit for each set of parameters tested.
                    By default this is set to be the sum of the residuals squared.
        '''
        self.x = np.array(x)
        self.y = np.array(y)
        self.model = model
        self.cost = cost
        self.walkers = []
        self.savepath=savepath
        if type(pnames) != type(None):
            self.pnames = list(pnames)

    def __getitem__(self, i):
        return self.walkers[i]

    def get_run_ids(self):
        return list(set([k for w in self.walkers for k in w.get_runs().keys()]))

    def add_walkers(self, n=1, p0=None, psig=None):
        '''
        Method for adding additional walkers to use during fitting.
        

        n    - Optional. Number of walkers to add. Default it 1.

        p0   - Starting position of the new walkers. Default is to start
               walkers off at the current position of the mcmc, self.p

        psig - Starting stepsize of the new walkers. Default is to start
               walkers off at the current position of the mcmc, self.p
        '''

        if type(p0)==type(None):
            try:
                p0 = self.walkers[-1].p
            except IndexError:
                raise ValueError("Could not determine initial set of parameters. p0 must be given for the first walker.")
        if type(psig)==type(None):
            try:
                psig = self.walkers[-1].psig
            except IndexError:
                pass

        for i in range(n):
            self.walkers.append(walker(self.x, self.y, self.model, p0, psig, self.cost))

    def get_best_params(self, run_id, method="mean"):
        if method=="mean":
            return np.mean(self.get_p_accepted(run_id)[:,1:], axis=0)
        if method=="best":
            p = self.get_p_accepted(run_id)
            rsqrd = p[:,0]
            p_accepted = p[:,1:]
            return p_accepted[np.argmin(rsqrd)]
        


    def move_to_best_walker(self, **kwargs):
        best = np.array([w.get_best_p(**kwargs) for w in self.walkers])
        best_params = best[:,1:]
        best_costs = best[:,0]
        best_wi = np.argmin(best_costs)
        for wi in range(len(self.walkers)):
            self.walkers[wi].p = best_params[best_wi]
            self.walkers[wi].psig = self.walkers[best_wi].psig.copy()
        self.p = best_params[best_wi]

    def save_walker_history(self, savepath=None, run_ids=None):
        if savepath == None:
            savepath=self.savepath
        if savepath == None:
            raise ValueError("Save path not specified.")

        #Check if run_id is a list or a single string
        #Make sure it's iterable.
        try:
            assert type(run_ids) == str
            run_ids = [run_ids]
        except AssertionError:
            pass
        try:
            iter(run_ids)
        except:
            run_ids = self.get_run_ids()
        
        for run_id in run_ids:
            for wi in range(len(self.walkers)):
                fmt = "%.5e" #["%d"]+["%.5e"]*(len(save_arr[0])-1)
                np.savetxt(savepath+"/"+run_id+"_"+str(wi)+".dat", self.walkers[wi].runs[run_id], fmt=fmt)
    
    def burn(self, nsteps):
        '''
        Method to run a burn stage. All walkers burn for the given number of steps.

        nsteps - Number of steps in the burn stage.
        '''
        for w in self.walkers:
            w.burn(nsteps)

    def check_convergence(self, tol):
        walker_means = np.vstack([w.get_mean() for w in self.walkers])
        mean = np.mean(walker_means, axis=0)
        #print np.max(np.abs(walker_means - mean), axis=0)
        #print mean, np.std(walker_means, axis=0)
        return np.all( np.abs(walker_means - mean) < tol )

    def walk(self, nsteps, wi='all', run_id=None, save=True):#tol, min_nsteps=1000, max_nsteps=2000):
        if type(wi)==str and wi=='all':
            wi = range(len(self.walkers))
        for n in range(nsteps):
            #Walk
            for i in wi:
                self.walkers[i].walk(1, run_id=run_id)

            #Update walker history.
            if save and self.savepath != None and n%10==0:
                self.save_walker_history(run_ids=run_id)

    def get_p_accepted(self, run_id):
        return np.vstack([w.runs[run_id] for w in self.walkers if run_id in w.runs])

    def plot_accepted(self, run_id):
        axes = [plt.subplots()[1] for n in range(int(comb(len(self.walkers[0].p),2)))]
        for i,w in enumerate(self.walkers):
            w.plot_accepted(run_id, axes=axes, label="Walker #"+str(i+1))

    def corner(self, run_id, p_crosshair=None, **kwargs):
        #Update parameters for creating contour plots.
        contour_kwargs = {
                'bins':35,
                'threshold':2,
                'marker_color':None,
                'ncontours':5,
                'fill':True,
                'mesh':False,
                'mesh_alpha':0.5,
                'cmap':None}
        contour_kwargs.update(kwargs)

        #Determine points to draw crosshair at.
        if type(p_crosshair) == str:
            method = p_crosshair
            p_crosshair = self.get_best_params(run_id, method=method)
        
        #Create figure
        fig = plt.figure()
        p_accepted = self.get_p_accepted(run_id)[:,1:]
        nparams = p_accepted.shape[1]

        # Draw diagonal histograms
        diag_axes = []
        for p in range(nparams):
            ax = fig.add_subplot(nparams, nparams, 1 + p*(nparams+1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            bins, edges = np.histogram(p_accepted[:,p], 30)
            xlow, xhigh = edges[:-1],edges[1:]
            x = np.array([xlow, xhigh]).T.flatten()
            y = np.array([bins,bins]).T.flatten()
            ax.plot(x, y, color="black", lw=1)
            if type(p_crosshair) != type(None):
                ax.axvline(x=p_crosshair[p], color="black", lw=1)

            diag_axes.append(ax)


        # Draw 2-parameter contours
        #Choose 200 random points to plot as a scatter plot over the contours.
        p_scat = p_accepted[np.random.choice(len(p_accepted), min([len(p_accepted),200]), replace=False)]

        labels = ["Q"+str(i+1) for i in range(nparams)]

        ax_sharey = {}
        for p1 in range(nparams):
            for p2 in range(p1+1, nparams):
                if not p2 in ax_sharey:
                    ax = fig.add_subplot(nparams, nparams, 1 + p1 + p2*nparams, sharex=diag_axes[p1])
                    ax_sharey[p2] = ax
                else:
                    ax = fig.add_subplot(nparams, nparams, 1 + p1 + p2*nparams, sharex=diag_axes[p1], sharey=ax_sharey[p2])

                if p2 != nparams-1:
                    ax.get_xaxis().set_visible(False)
                else:
                    ax.set_xlabel(labels[p1])
                if p1 != 0:
                    ax.get_yaxis().set_visible(False)
                else:
                    ax.set_ylabel(labels[p2])
                contour(p_accepted[:,p1], p_accepted[:,p2], axis=ax, **contour_kwargs) 
                ax.scatter(p_scat[:,p1], p_scat[:,p2], s=1, color='black', alpha=0.2)
                if type(p_crosshair) != type(None):
                    ax.scatter(p_crosshair[p1], p_crosshair[p2], color="black", s=3)
                    ax.axvline(x=p_crosshair[p1], color="black", lw=1)
                    ax.axhline(y=p_crosshair[p2], color="black", lw=1)

        #Decrease spacing between axes.
        plt.subplots_adjust(wspace=0.05, hspace=0.06)

    def plot_sample(self, run_id, n):
        fig, ax = plt.subplots()
        p_accepted = self.get_p_accepted(run_id)[:,1:]
        x = np.linspace(min(self.x), max(self.x), 1000)
        for i in np.random.choice(len(p_accepted), n):
            ax.plot(x, self.model(x, p_accepted[i]), color="blue", alpha=1./n**0.65, zorder=0)
        ax.scatter(self.x, self.y, color="black")

    def load_walker_history(self, path):
        '''
        Function for loading walkers from previously saved walker histories.
        
        path - Path to the .mpac file to be loaded.
        '''
        walker_runs = {}
        for fpath in glob.glob(path+"/*.dat"):
            run_id, wi = (fpath.split("/")[-1])[:-4].split("_")
            if not wi in walker_runs:
                walker_runs[wi] = {}
            walker_runs[wi][run_id] = np.loadtxt(fpath)

        for k in sorted(walker_runs.keys(),key=lambda s: int(s)):
            sample_p = walker_runs[k].values()[0][0][1:]
            w = walker(self.x, self.y, self.model, np.ones_like(sample_p))
            w.runs = walker_runs[k]
            self.walkers.append(w)
