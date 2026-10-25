use libm::log2;
use rand::Rng;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_distr::{Normal, Distribution};
use statrs::distribution::{Normal as nm, Continuous as cn};
use weighted_rand::builder::*;
extern crate csv;
use std::error::Error;
use std::fs::OpenOptions;
use std::print;
use std::println;
use std::vec;
use rayon::prelude::*;
use std::env;
use std::fs::{self, File};
use std::path::Path;
use csv::Writer;
use std::io::BufWriter;

// R stuff
use std::process::{Command, Stdio};
use std::io::Write;
use std::sync::Mutex;

pub struct RSession {
    child: std::process::Child,
}

impl RSession {
    pub fn new() -> std::io::Result<Self> {
        let child = Command::new("R")
            .args(["--vanilla", "--quiet", "--slave"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;

        Ok(Self { child })
    }

    pub fn exec(&mut self, code: &str) -> std::io::Result<()> {
        let stdin = self.child.stdin.as_mut().unwrap();
        stdin.write_all(code.as_bytes())?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        Ok(())
    }
}

//Functions
fn write_csv_header(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file_path1 = format!("{}summaries.csv", path);

    let file = match File::create(&file_path1) {
        Ok(f) => f,
        Err(e) => return Err(Box::new(e)),
    };

    let buf_writer = BufWriter::new(file);
    let mut wtr = Writer::from_writer(buf_writer);

    wtr.write_record(&[
        "i","pop_len","stoch","theta_low","theta_high","sigma_mass","cli_sigma","days",
        "turns","obs_c","d","stability","mean_obs","climate_match","p","mean_fitness","ve","lambda",
        "psi","vs","nest_mass","mu_sig", "cue_error", "prior"
    ])?;

    wtr.flush()?;
    Ok(())
}

fn write_csv_header2(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file_path1 = format!("{}", path);

    let file = match File::create(&file_path1) {
        Ok(f) => f,
        Err(e) => return Err(Box::new(e)),
    };

    let buf_writer = BufWriter::new(file);
    let mut wtr = Writer::from_writer(buf_writer);

    wtr.write_record(&[
        "i","pop_len","stoch","theta_low","theta_high","sigma_mass","cli_sigma","days",
        "turns","obs_c","d","stability","mean_obs","climate_match","p","mean_fitness","ve","lambda",
        "psi","vs","nest_mass","mu_sig", "cue_error", "prior"
    ])?;

    wtr.flush()?;
    Ok(())
}

fn try_print(vec:Vec<f64>, file:&str) -> Result<(), Box<dyn Error>> {
    let strings: Vec<String> = vec.iter().map(|n| n.to_string()).collect();

    let outfile = OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(file)
        .unwrap();
    let mut wtr = csv::Writer::from_writer(outfile);
    wtr.write_record(strings)?;
    wtr.flush()?;
    Ok(())
}

// Structs
const NS: usize = 6;
const NPI: usize = 6;
const NM: usize = 6;
const I: usize = 3;
const G: usize = 5;
const LENPI: usize = (NPI-1+1)*(NPI-1+2)/2;
const N_states: usize = NS * NM * LENPI;

#[derive(Clone, Debug)]
struct Environment {
    hi: Vec<f64>, //(i) pdf over patch population densities
    hi_prime: Vec<f64>, //(i) pdf over patch population densities post some event
    n_max: f64,
    g_mean: f64,
    g_sd: f64,
    m_switch: usize, //threshold m above which fighter phenotype is expressed
    r_fighter: f64,
    r_m: f64, // benefit of quality
    r_baseline: f64,
    a_qi: Vec<f64>, //(i) arrivals from dispersal
    mu: f64, // Dispersal rate
    r_qi: Vec<f64>, //(i) arrivals from local births
    R: Vec<Vec<Vec<f64>>>, //(s,m,i) fecundity of individual of size s and maturity m
    rhoi: Vec<Vec<f64>>, //(idx,i) prob. distribution over states for individuals in patches of density qi
    p_births: Vec<Vec<f64>>, //(i,j)
    rhotilde: Vec<f64>, //(idx)
    p_g_i: Vec<Vec<f64>>,
    pipig: Vec<Vec<Vec<f64>>>, //Trans. prob. og pi prime given pi and g [piprime][pi][g]
    beliefs: Vec<BeliefState>,
    x: Vec<usize>, // (idx) gives resident strategy -> threshold value above which input energy goes to quality
    dev_m: Vec<Vec<Vec<Vec<f64>>>>, // (idx prime, idx, g, kappa) developmental transitions
    y_max:f64, // maximum maintenance investment
    survival: Vec<Vec<f64>>, // ((s,pi,m),g) prob. survival
    lambda: f64, // Baseline rate of death
    rhohat: Vec<Vec<f64>>, // state distribution after development fatalities, before normalisation
    pfail: Vec<Vec<f64>>, // prob of death for current physiological and habitat state over development
    sigma_per: f64, // standard deviation of patch quality perturbations
    p_tau: Vec<f64>, // pseudonormal probability distribution over perturbations
    gamma: f64, // parameter determining maximum effect of demographic feedback on patch survival
    c: Vec<usize>, // (i) rounded shift in prob. of posterior patch state after perturbatins given pop. density i
    tol: f64, // tolerance threshold for difference in demographic state, below which projections end
    p_pert: Vec<Vec<f64>>, // transition probabilities of patches given perturbations
    x_prime: Vec<usize>, // Best response strategy
    ptilde: Vec<Vec<f64>>, // (i_prime,i) Evolved prior for transition prob. of patches at demographic stability
    V: Vec<Vec<f64>>, // (state,n) Fitness
    tolv: f64, // tolerance threshold in the difference in v for finding the best response
    pipitime: Vec<Vec<f64>>, // (piprime,pi) prob of belief transition given time
    p_g_pi: Vec<Vec<f64>>, // (g,pi) the probability of food mass g given prior pi
    pert_survival: Vec<f64>, // (pi) probability of survival given prior belief pi
    markov: Vec<Vec<Vec<f64>>>, //(state,stateprime,kappa) transition prob given action kappa
}

#[derive(Clone, Debug)]
struct BeliefState {
    pi0: f64,
    pi1: f64,
    pi2: f64,
}

// Implementations
fn bernouli(prob:f64) -> f64 {
    let dice:f64 = rand::rng().random();
    if dice < prob { 1.0 } else { 0.0 }
}

fn idx(s:usize, pi:usize, m:usize) -> usize {
    // returns index integer from individual state
    s * LENPI * NM + pi * NM + m
}

fn vectorise_pi(n0:usize,n1:usize)->usize{
    // first term tells you which row (n0) if it were a 2x2 grid, 
    // second term accounts for missing lower triangle given simplex
    // third term gives the column (n1)
    if n0==0 { n1 } else { n0*(NPI-1) - (n0*(n0-1))/2 + n1 }
}

impl Environment { 
    fn init_beliefs(&mut self){
        // Run at the beginning to initialise the vector of belief states
        for n0 in 0..NPI{
            for n1 in 0..NPI{
                if NPI-1 >= n0+n1{
                    let j = vectorise_pi(n0, n1);
                    println!("{}, {}, {}",n0,n1,j);
                    self.beliefs[j].pi0=(n0 as f64)/(NPI as f64-1.);
                    self.beliefs[j].pi1=(n1 as f64)/(NPI as f64-1.);
                    self.beliefs[j].pi2=((NPI-n0-n1-1) as f64)/(NPI as f64-1.);
                    println!("{}, {}, {}",self.beliefs[j].pi0, self.beliefs[j].pi1, self.beliefs[j].pi2);
                }
            }
        }
    }

    fn q(&self, i: usize)->f64{
        return self.n_max*(i as f64 + 1.)/(I as f64)
    }

    fn interpolate_i(&self, mut prime:f64, min_val: f64, max_val: f64) -> (usize,usize,f64,f64) {
        prime = prime.min(max_val).max(min_val);
        let d1 = ((I as f64 -1.) * prime/self.n_max).floor();
        let d2 = (d1 + 1.0).min(2.);
        let p2 = ((I as f64 -1.) * prime/self.n_max) - d1;
        let p1 = 1.0 - p2;
        println!("{}, {}", d1, d2);

        return (d1 as usize,d2 as usize,p1,p2)
    }

    fn decompose_pi(&self, prime1:f64, prime2:f64, prime3:f64) -> (usize,usize,usize,f64,f64,f64) {
        // Interpolation step: decompose into interger and fractional parts
        let d1 = ((NPI as f64) * prime1).floor();
        let d2 = ((NPI as f64) * prime2).floor();
        let d3 = ((NPI as f64) * prime3).floor();

        let p1 = (NPI as f64) * prime1 - d1;
        let p2 = (NPI as f64) * prime2 - d2;
        let p3 = (NPI as f64) * prime3 - d3;

        return (d1 as usize, d2 as usize, d3 as usize, p1, p2, p3)
    }

    fn init_R(&mut self){
        let mut r:f64;
        for i in 0..I{
            // Sum of g
            let sum_g:f64 = (0..G)
                .map(|g| self.p_g_i[g][i] * g as f64)
                .sum();

            for m in 0..NM{
                for s in 0..NS{

                    r = 0.;
                    
                    if s >= NS {
                        if m > self.m_switch{
                            r = sum_g * (self.r_baseline + (m as f64) *self.r_m + self.r_fighter);
                        } else {
                            
                            r = sum_g * (self.r_baseline + (m as f64) *self.r_m);
                        
                        }
                    }

                    self.R[s][m][i] = r;

                }
            }
        }
    }

    fn update_r(&mut self){
        for i in 0..I {
            self.r_qi[i] = 0.0;

            for s in 0..NS{
                for pi in 0..LENPI{
                    for m in 0..NM{
                        self.r_qi[i] += self.rhoi[idx(s,pi,m)][i]*self.R[s][m][i]
                    }
                }
            }
        }
    }

    fn interpolate_pi(&self, pi0: f64, pi1: f64, pi2: f64) -> (usize,usize,usize,f64,f64,f64){
        
        // interpolating priors onto the grid
        let (d1, d2, d3, f1, f2, f3) = self.decompose_pi(pi0,pi1,pi2);
        let (mut coord0,mut coord1,mut coord2,mut p0,mut p1,mut p2)=(0,0,0,0.,0.,0.);
        if (f1+f2+f3) as usize == 0{

            coord0 = vectorise_pi(d1, d2);
            p0 = 1.;

        } else if (f1+f2+f3) as usize == 1{

            coord0 = vectorise_pi(d1+1, d2);
            p0 += f1;

            coord1 = vectorise_pi(d1, d2+1);
            p1 += f2;

            coord2 = vectorise_pi(d1, d2);
            p2 += f3;
            
        } else if (f1+f2+f3) as usize == 2{

            coord0 = vectorise_pi(d1, d2+1);
            p0 += 1.-f1;

            coord1 = vectorise_pi(d1+1, d2);
            p1 += 1.-f2;

            coord2 = vectorise_pi(d1+1, d2+1);
            p2 += 1.-f3;
 
        }
        return (coord0,coord1,coord2,p0,p1,p2)
    }

    fn newborns(&mut self){
        self.rhotilde.fill(0.0);

        // interpolating priors onto the grid
        let (pi0_prior, pi1_prior, pi2_prior, p0, p1, p2) = self.interpolate_pi(self.hi[0],self.hi[1],self.hi[2]);

        // Add to starting line
        self.rhotilde[idx(0,pi0_prior,0)] += p0;
        self.rhotilde[idx(0,pi1_prior,0)] += p1;
        self.rhotilde[idx(0,pi2_prior,0)] += p2;
    }

    fn Dispersal(&mut self){
        // This function projects the distributions h_qi and rho given newborns and dispersal
        // as per section 3.2 Dispersal
        // First we will define the patch density transition probabilities, as per section 3.2.2
        // Intermediate variables include alpha "a_qi", the dispersing pool, and r_qi
        
        // first we update r_qi, as per equation 5 in 3.2.2:
        self.update_r();

        // Define newborn physiology
        self.newborns();

        // Set rho prime to zero so we can add to it.
        let mut rhoiprime: Vec<Vec<f64>> = vec![vec![0.0;I]; N_states];

        // 3.2.2 Patch projection
        // Second expression of equation 4 defining alpha, representing the number of migrants
        let dispersing_pool: f64 = (0..I)
                .map(|j| self.hi[j]*self.q(j)*self.r_qi[j])
                .sum();

        for i in 0..I {
            // calculating alpha, equation 4
            self.a_qi[i] = (1. - self.mu)*self.q(i)*self.r_qi[i] + self.mu*dispersing_pool;

            // calculating the new patch density size after dispersal, equation 6
            let qiprime: f64 = self.a_qi[i] + self.q(i);

            // Interpolating iprime to find transition probabilities on 1-simplex qi grid, equations 7-10
            let (d1, d2, p1, p2) = self.interpolate_i(qiprime, 0.0, self.q(I));
            self.p_births[i].fill(0.0); 
            self.p_births[i][d1] += p1;
            self.p_births[i][d2] += p2;
        }
            
        // Now we move on to projection of the patch density distribution 
        // and of physiological states:
        for i in 0..I {
            // Physiological state distribution update
            for s in 0..NS {
                for pi in 0..LENPI {
                    for m in 0..NM {
                        for j in 0..I{
                            rhoiprime[idx(s,pi,m)][i] += self.hi[j]*self.p_births[j][i]*((self.q(j)/self.q(i))*self.rhoi[idx(s,pi,m)][j] + ((self.q(i)-self.q(j))/self.q(i))*self.rhotilde[idx(s,pi,m)]);
                            
                            if rhoiprime[idx(s,pi,m)][i].is_nan(){
                                println!("rhoiprime: {}, {}, {}, {}, {}, {}, {}", rhoiprime[idx(s,pi,m)][i], self.hi[j], self.p_births[j][i], (self.q(j)/self.q(i)), self.rhoi[idx(s,pi,m)][j], ((self.q(i)-self.q(j))/self.q(i)), self.rhotilde[idx(s,pi,m)]);
                            }
                        }
                    }
                }
            }
        }
        for i in 0..I {
            // make rhoi rhoiprime
            for state in 0..N_states {
                self.rhoi[state][i] = rhoiprime[state][i];
            }
        }

        // Patch state distribution update
        let mut h_prime: Vec<f64> = vec![0.0;I];
        for j in 0..I {
            for i in 0..I {
                h_prime[i] += self.hi[j] * self.p_births[j][i]; // Equation 13
            }
        }
        for i in 0..I {
            self.hi[i] = h_prime[i];
        }

    }

    fn init_p_g_i(&mut self){
        // This function updates the probability of a food mass of g in a patch of density i
        // as per the first equaitions of section 3.3 defining the pdf over food masses.
        let mut q_i:f64;
        for i in 0..I{
            q_i = self.q(i);
            // Equation 18
            let K: f64 = (0..G)
                .map(|g| (-((g as f64-self.g_mean/q_i).powf(2.))/(2.*self.g_sd.powf(2.))).exp())
                .sum();

            // Equation 17
            for g in 0..G{
                self.p_g_i[g][i] = (1./K) * (-((g as f64-self.g_mean/q_i).powf(2.))/(2.*self.g_sd.powf(2.))).exp();
            }
        }
    }

    fn init_pipig(&mut self){
        // This function defines the transition probabilities of beliefs
        // for each given food mass of g
        let mut pi1;
        let mut pi2;
        let mut pi3;
        let mut prime1;
        let mut prime2;
        let mut prime3;
        for vec in &mut self.pipig {
            for vec2 in vec {
                vec2.fill(0.);
            }
        }
        for pi in 0..LENPI{
            for g in 0..G{
                // extract belief distribtion from flat pi vertex index
                pi1 = self.beliefs[pi].pi0;
                pi2 = self.beliefs[pi].pi1;
                pi3 = self.beliefs[pi].pi2;

                // Find posterior belief distribution
                prime1 = (pi1 * self.p_g_i[g][0])/(pi1 * self.p_g_i[g][0] + pi2 * self.p_g_i[g][1] + pi3 * self.p_g_i[g][2]);
                prime2 = (pi2 * self.p_g_i[g][1])/(pi1 * self.p_g_i[g][0] + pi2 * self.p_g_i[g][1] + pi3 * self.p_g_i[g][2]);
                prime3 = (pi3 * self.p_g_i[g][2])/(pi1 * self.p_g_i[g][0] + pi2 * self.p_g_i[g][1] + pi3 * self.p_g_i[g][2]);

                // Find fractional parts
                let (n1, n2, n3, p1, p2, p3) = self.decompose_pi(prime1, prime2, prime3);
                
                // Interpolate on the 2-Simplex and flatten
                if (p1+p2+p3)>1. {
                    self.pipig[vectorise_pi(n1, n2+1)][pi][g] += p1;
                    self.pipig[vectorise_pi(n1+1, n2)][pi][g] += p2;
                    self.pipig[vectorise_pi(n1+1, n2+1)][pi][g] += p3;
                } else if (p1+p2+p3)>0. {
                    self.pipig[vectorise_pi(n1+1, n2)][pi][g] += p1;
                    self.pipig[vectorise_pi(n1, n2+1)][pi][g] += p2;
                    self.pipig[vectorise_pi(n1, n2)][pi][g] += p3;
                } else {
                    self.pipig[vectorise_pi(n1, n2)][pi][g] += 1.;
                }
            }
        }
    }

    fn y(&self, s:f64, m:f64) -> f64{
        let s3 = s*s*s;
        let max_s3 = (NS as f64).powf(3.);
        return (s3*m/(max_s3 * NM as f64))*self.y_max
    }

    fn interpolate_m(&self, mut post_m:f64) -> (usize,f64,usize,f64){
        post_m = post_m.min((NM-1) as f64);
        let md1 = (post_m).floor();
        let mp2 = post_m-md1;
        let md2 = (md1+1.).min((NM-1) as f64);
        let mp1 = 1.-mp2;

        return (md1 as usize, mp1, md2 as usize, mp2)
    }

    fn interpolate_s(&self, mut post_s:f64) -> (usize,f64,usize,f64){
        post_s = post_s.min((NS-1) as f64);
        let sd1 = (post_s).floor();
        let sp2 = post_s-sd1;
        let sd2 = (sd1+1.).min((NS-1) as f64);
        let sp1 = 1.-sp2;

        return (sd1 as usize, sp1, sd2 as usize, sp2)
    }

    fn update_dev_m(&mut self){
        // Udates the state transition matrix, dev_M, given the current resident strategy
        
        // Set developmental transition matrix to equal 0 for all prior and posterior states
        for prior in self.dev_m.iter_mut() {
            for post in prior.iter_mut() {
                for g in post.iter_mut(){
                    g.fill(0.); // for each prior and post states (s and s'), over all g: D(s'|s,g)=0
                }
            }
        }

        let mut del_m;
        let mut del_s;
        // for each prior state
        for s in 0..NS {
            for pi in 0..LENPI {
                for m in 0..NM {
                    // for each food realisation
                    for g in 0..G {
                        // After development:
                        del_m = (g as f64-self.x[idx(s,pi,m)] as f64).max(0.);
                        del_s = (g as f64-del_m-self.y(s as f64,m as f64)).max(0.);
                        
                        // Maintenance survival probs section 3.3.3 Maintenance
                        self.survival[idx(s,pi,m)][g] = ((g as f64-del_m-del_s)/self.y(s as f64,m as f64)).min(1.);

                        // Interpolate onto s and m grids:
                        let (md1,mp1,md2,mp2) = self.interpolate_m(del_m+(m as f64));
                        let (sd1,sp1,sd2,sp2) = self.interpolate_s(del_s+(s as f64));

                        // Developmental state transitions 
                        self.dev_m[idx(sd1, pi, md1)][idx(s, pi, m)][g][self.x[idx(s,pi,m)]]=mp1*sp1;
                        self.dev_m[idx(sd2, pi, md1)][idx(s, pi, m)][g][self.x[idx(s,pi,m)]]=mp1*sp2;
                        self.dev_m[idx(sd1, pi, md2)][idx(s, pi, m)][g][self.x[idx(s,pi,m)]]=mp2*sp1;
                        self.dev_m[idx(sd2, pi, md2)][idx(s, pi, m)][g][self.x[idx(s,pi,m)]]=mp2*sp2;
                        
                    }
                }
            }
        }
    }

    fn state_invert(&self, state:usize) -> (usize,usize,usize){
        let m = state % NS;
        let pi = ((state-m)/NS) % LENPI;
        let s = ((state-m)/(NS)-pi) / LENPI;
        return (s,pi,m)
    }

    fn Foraging(&mut self){
        // This function will do all the updates associates with foraging and development
        // as per section 3.3 in finding demographic stability.

        // Development, section 3.3.2
        self.update_dev_m();

        // update state distributions, section 3.3.4, rho hat prime
                // Given all the update transition probabilities and the prior distribution of states
        // update the distribution of states after foraging
        let (mut s,mut pi,mut m,mut s_prime,mut pi_prime,mut m_prime);
        for i in 0..I{
            for state_prime in 0..N_states{
                // set post to zero
                self.rhohat[state_prime][i] = 0.;
                (s_prime,pi_prime,m_prime) = self.state_invert(state_prime);

                        
                        
                for state in 0..N_states{
                    (s,pi,m) = self.state_invert(state);
                    for g in 0..G{
                        self.rhohat[state_prime][i] += 
                            self.dev_m[state_prime][state][g][self.x[idx(s,pi,m)]] 
                                * self.survival[idx(s,pi_prime,m)][g] 
                                * self.pipig[pi_prime][pi][g] 
                                * self.p_g_i[g][i] 
                                * self.rhoi[idx(s, pi, m)][i];
                        
                        // println!("devm: {}, surv:{}, pipig: {}, pgi: {}, rhoi: {}", self.dev_m[state_prime][state][g][self.x[idx(s,pi,m)]],
                        //     self.survival[idx(s,pi_prime,m)][g],
                        //     self.pipig[pi_prime][pi][g],
                        //     self.p_g_i[g][i],
                        //     self.rhoi[idx(s, pi, m)][i]
                        // );
                    }
                }
            }
        }

        // Update the frequency distribution of patches over patch density states
        // clear pfail
        for vec in &mut self.pfail{
            vec.fill(0.);
        }

        // update pfail
        for i in 0..I{
            // calculate u, the posterior population density of patches previously of state i
            let rhohatsum: f64 = (0..N_states)
                .map(|state| self.rhohat[state][i])
                .sum();
            let u = self.q(i) * (1. - self.lambda) * rhohatsum;

            println!("rhohatsum: {}, u:{}", rhohatsum, u);
            // interpolate u
            let u1 = (I as f64 * u/self.n_max).floor();
            let u2 = u1+1.;
            let p2 = (I as f64 * u/self.n_max) - u1;
            let p1 = 1. - p2;

            // algorithm to update pfail, the transition probabilities between patch states
            self.pfail[u1 as usize][i] += p1; 
            self.pfail[u2 as usize][i] += p2; 

            // println!("p0: {}, p1: {}", p1, p2); // confirmed sum to 1.
            
        }
        
        // println!("hi priors: {}, {}, {}",  self.hi[0],  self.hi[1],  self.hi[2]);
        // update h_i
        self.hi_prime.fill(0.);
        for i in 0..I{
            for i_prime in 0..I{
                self.hi_prime[i_prime] += self.hi[i] * self.pfail[i_prime][i];
            }
        }
        // println!("hi primes: {}, {}, {}",  self.hi_prime[0],  self.hi_prime[1],  self.hi_prime[2]);

        let mut total = vec![0.;I];
        for i in 0..I{
            // equation 30, rho prime prime
            for state in 0..N_states{
                total[i] += self.rhohat[state][i];
            }
        }

        for i_prime in 0..I{
            for state_prime in 0..N_states{
                self.rhoi[state_prime][i_prime] = 0.;
                
                for i in 0..I{
                    self.rhoi[state_prime][i_prime] += self.hi[i] * self.pfail[i_prime][i] * self.rhohat[state_prime][i]/total[i];
                }
            }
        }
            
        self.hi = self.hi_prime.clone();
        
    }

    fn init_ptau(&mut self) {
        // Init pseudonormal distribution:
        // let k: f64 = (0..(2*I))
        //         .map(|tau| (-((tau as i32-I as i32) as f64/I as f64).powf(2.)/(2.*self.sigma_per*self.sigma_per)).exp())
        //         .sum();
        // for tau in 0..(2*I){
        //     self.p_tau[tau] = (1./k) * (-((tau as i32-I as i32) as f64/I as f64).powf(2.)/(2.*self.sigma_per*self.sigma_per)).exp();
        // }

        // // Define shift given demographic feedback
        // for j in 0..I{
        //     let c = (self.gamma * (j as f64/I as f64)) as usize;
        //     for i in 0..I{
        //         if i == 0 {
        //             self.p_pert[i][j] = (0..((i+c).saturating_sub(j)).min(2*I))
        //                 .map(|tau| self.p_tau[tau])
        //                 .sum();

        //         } else if i == I {
        //             self.p_pert[i][j] = self.p_tau[((i+c).saturating_sub(j)).min(2*I)];

        //         } else {
        //             self.p_pert[i][j] = (((i+c).saturating_sub(j)).min(2*I)..2*I)
        //                 .map(|tau| self.p_tau[tau])
        //                 .sum();
        //         }
        //     }
        // }
        self.p_pert[0][0] = 0.5;
        self.p_pert[1][0] = 0.5;
        self.p_pert[2][0] = 0.0;
        self.p_pert[0][1] = 1./3.;
        self.p_pert[1][1] = 1./3.;
        self.p_pert[2][1] = 1./3.;
        self.p_pert[0][2] = 0.25;
        self.p_pert[1][2] = 0.5;
        self.p_pert[2][2] = 0.25;
    }

    fn Perturbations(&mut self){
        // section 3.4, this function updates the patch distribution given quality perturbations
        // Update hi_prime
        self.hi_prime.fill(0.);
        for i_prime in 0..I{
            self.hi_prime[i_prime] = (0..I)
                .map(|i| self.p_pert[i_prime][i] * self.hi[i])
                .sum();
        }

        // update rhoi
        for i_prime in 0..I{
            for state in 0..N_states{
                self.rhohat[state][i_prime] = (0..I)
                    .map(|i| self.p_pert[i_prime][i] * self.hi[i] * self.rhoi[state][i])
                    .sum();
            }
        }
        self.rhoi = self.rhohat.clone();
        self.hi = self.hi_prime.clone();
    }

    fn bayes(&mut self) {
        let mut rhoi_prime: Vec<Vec<f64>> = vec![vec![0.0;I]; N_states];
        for state in 0..N_states{
            for i in 0..I{
                // retrieve state
                let (s,pi,m) = self.state_invert(state);
                let (pi0,pi1,pi2) = (self.beliefs[pi].pi0, self.beliefs[pi].pi1, self.beliefs[pi].pi2);

                // find continuous posterior values
                // ptilde is the transition probability of patches at demographic stability
                // ptilde just equals 0 on the first iteration
                let post0 = pi0 * self.ptilde[0][0] + pi1 * self.ptilde[0][1] + pi2 * self.ptilde[0][2];
                let post1 = pi0 * self.ptilde[1][0] + pi1 * self.ptilde[1][1] + pi2 * self.ptilde[1][2];
                let post2 = pi0 * self.ptilde[2][0] + pi1 * self.ptilde[2][1] + pi2 * self.ptilde[2][2];

                // interpolate onto the 2-Simplex
                let (pi0_post, pi1_post, pi2_post, p0, p1, p2) = self.interpolate_pi(post0, post1, post2);

                // update rhoi
                rhoi_prime[idx(s, pi0_post, m)][i] += self.rhoi[idx(s, pi, m)][i] * p0;
                rhoi_prime[idx(s, pi1_post, m)][i] += self.rhoi[idx(s, pi, m)][i] * p1;
                rhoi_prime[idx(s, pi2_post, m)][i] += self.rhoi[idx(s, pi, m)][i] * p2;
            }
        }
        self.rhoi = rhoi_prime;
    }

    fn Dem_Dif(&self, oldh:Vec<f64>) -> f64 {
        let dif = (0..I)
            .map(|i| (&self.hi[i] - oldh[i]).abs())
            .sum();
        return dif
    }

    fn init_h(&mut self){
        // starting point for the distribution of patch densities
        self.hi[0] = 1./3.;
        self.hi[1] = 1./3.;
        self.hi[2] = 1./3.;
    }

    fn init_rho(&mut self){
        for i in 0..I{
            self.rhoi[idx(0, LENPI/2, 0)][i] = 1.0;
        }
    }

    fn inits(&mut self){
        println!("Initialising h");
        self.init_h();

        println!("Initialising rho");
        self.init_rho();

        println!("Initialising beliefs");
        self.init_beliefs();

        // as per section 3.3 in finding demographic stability.
        println!("Initialising p(g|i)");
        // first update food mass pdf:
        self.init_p_g_i();

        // Initialise the transition probabilities from pi to pi^prime given a food mass of g
        println!("Initialising P(pi|pi,g)");
        self.init_pipig();

        // Init pseudonormal distribution and define perturbations (p_pert(i|j)):
        println!("Initialising P(tau)");
        self.init_ptau();

        // init reproductive rates
        println!("Initialising R");
        self.init_R();

        // init prob of g given prior pi
        println!("Initialising P(g|pi)");
        self.init_gpi();

        // init prob of survival given perturbations
        println!("Initialising P(survival|pert");
        self.init_pertsurvival();
        
        println!("Initialisations done!");
    }

    fn r_state(&self, state:usize) -> f64{
        let (s,pi,m) = self.state_invert(state);
        return self.R[s][m][0]*self.beliefs[pi].pi0 + self.R[s][m][1]*self.beliefs[pi].pi1 + self.R[s][m][2]*self.beliefs[pi].pi2;
    }

    fn update_dev_m_kappa(&mut self){
        
        // Udates the state transition matrix, dev_M, given all possible actions
        
        // Set developmental transition matrix to equal 0 for all prior and posterior states
        for prior in self.dev_m.iter_mut() {
            for post in prior.iter_mut() {
                for g in post.iter_mut(){
                    g.fill(0.); // for each prior and post states (s and s'), over all g: D(s'|s,g)=0
                }
            }
        }

        let mut del_m;
        let mut del_s;
        // for each prior state
        for s in 0..NS {
            for pi in 0..LENPI {
                for m in 0..NM {
                    // for each food realisation
                    for g in 0..G {
                        for k in 0..G{

                        // After development:
                        del_m = (g as f64-k as f64).max(0.);
                        del_s = (g as f64-del_m-self.y(s as f64,m as f64)).max(0.);
                        
                        // Maintenance survival probs section 3.3.3 Maintenance
                        self.survival[idx(s,pi,m)][g] = ((g as f64-del_m-del_s)/self.y(s as f64,m as f64)).min(1.);

                        // Interpolate onto s and m grids:
                        let (md1,mp1,md2,mp2) = self.interpolate_m(del_m+(m as f64));
                        let (sd1,sp1,sd2,sp2) = self.interpolate_s(del_s+(s as f64));

                        // Developmental state transitions 
                        self.dev_m[idx(sd1, pi, md1)][idx(s, pi, m)][g][k]=mp1*sp1;
                        self.dev_m[idx(sd2, pi, md1)][idx(s, pi, m)][g][k]=mp1*sp2;
                        self.dev_m[idx(sd1, pi, md2)][idx(s, pi, m)][g][k]=mp2*sp1;
                        self.dev_m[idx(sd2, pi, md2)][idx(s, pi, m)][g][k]=mp2*sp2;
                        }
                        
                    }
                }
            }
        }
    }

    fn update_pipitime(&mut self){
        // what is the probability of posterior belief pi prime 
        // given prior pi and the passing of time

        for pi in &mut self.pipitime{
            pi.fill(0.);
        }

        for pi in 0..LENPI{
            let (pi0, pi1, pi2) = (self.beliefs[pi].pi0, self.beliefs[pi].pi1, self.beliefs[pi].pi2);
            let trans1:f64 = (0..3)
                .map(|j| self.p_births[0][j]*self.p_pert[j][0])
                .sum();
            let trans2:f64 = (0..3)
                .map(|j| self.p_births[0][j]*self.p_pert[j][1])
                .sum();
            let trans3:f64 = (0..3)
                .map(|j| self.p_births[0][j]*self.p_pert[j][2])
                .sum();
            let trans4:f64 = (0..3)
                .map(|j| self.p_births[1][j]*self.p_pert[j][0])
                .sum();
            let trans5:f64 = (0..3)
                .map(|j| self.p_births[1][j]*self.p_pert[j][1])
                .sum();
            let trans6:f64 = (0..3)
                .map(|j| self.p_births[1][j]*self.p_pert[j][2])
                .sum();
            let trans7:f64 = (0..3)
                .map(|j| self.p_births[2][j]*self.p_pert[j][0])
                .sum();
            let trans8:f64 = (0..3)
                .map(|j| self.p_births[2][j]*self.p_pert[j][1])
                .sum();
            let trans9:f64 = (0..3)
                .map(|j| self.p_births[2][j]*self.p_pert[j][2])
                .sum();
            let post0 = pi0*trans1 +pi1*trans2 + pi2*trans3;
            let post1 = pi0*trans4 +pi1*trans5 + pi2*trans6;
            let post2 = pi0*trans7 +pi1*trans8 + pi2*trans9;

            let (d1, d2, d3, p1, p2, p3) = self.interpolate_pi(post0, post1, post2); 
            self.pipitime[d1][pi] += p1;
            self.pipitime[d2][pi] += p2;
            self.pipitime[d3][pi] += p3;
        }
    }

    fn demog(&mut self){
        // demographic stability:
        let mut dif_h = 1.;
        let mut oldh: Vec<f64> = self.hi.clone();
        while dif_h > self.tol{ 
        }
        self.update_dev_m_kappa();
        self.update_pipitime();
        println!("Demog done: h(0): {}, h(1): {}, h(2): {}", self.hi[0], self.hi[1], self.hi[2]);

    }

    fn difv(&self) -> f64 {
        let tot:f64 = (0..N_states)
            .map(|state| (self.V[0][state] - self.V[1][state]).abs())
            .sum();
        return tot
    }

    fn init_gpi(&mut self){
        for pi in 0..LENPI{
            for g in 0..G{
                let belief = &self.beliefs[pi];
                let sum = self.p_g_i[g][0]*belief.pi0 + self.p_g_i[g][0]*belief.pi1 + self.p_g_i[g][0]*belief.pi2;
                self.p_g_pi[g][pi] = sum;
            }
        }
    }

    fn init_pertsurvival(&mut self){
        for pi in 0..LENPI{
            let belief = &self.beliefs[pi];
            let mut sumjk = 0.;
            for k in 0..I{
                sumjk += belief.pi0*(1. - self.p_pert[k][0])*((self.q(0)-self.q(k))/self.q(0)); 
                sumjk += belief.pi1*(1. - self.p_pert[k][1])*((self.q(1)-self.q(k))/self.q(1)); 
                sumjk += belief.pi2*(1. - self.p_pert[k][2])*((self.q(2)-self.q(k))/self.q(2)); 
            }
            self.pert_survival[pi] = (1. - self.lambda) * sumjk;
        }
    }

    fn update_markovian(&mut self){
        // Update markovian transition matrix, each cell representing the
        // probability of an individual surviving and transitioning from state
        // state -> state_prime, given strategy k at demographic stability with 
        // with resident strategy x(s,pi,m).
        let mut p:f64 = 0.;

        // find the transition probability from state to state_prime
        for state in 0..N_states{
            let (s,pi,m) = self.state_invert(state);
            for state_pripri in 0..N_states{
                let (s_pripri,pi_pripri,m_pripri) = self.state_invert(state_pripri);
                
                // With the probability dependent on the chosen developmental action k=kappa
                for k in 0..G{

                    // Sum over all belief states given updating from time
                    let mut sum_piprime: f64 = 0.;
                    for piprime in 0..LENPI{
                        // sum over all possible food masses (g)
                        sum_piprime += (0..G)
                            .map(|g| self.pipitime[piprime][pi]
                                * self.p_g_pi[g][piprime]
                                * self.pipig[pi_pripri][piprime][g]
                                * self.dev_m[state_pripri][idx(s, pi_pripri, m)][g][k])
                            .sum::<f64>();
                    }
                    
                    // with probability of death from perturbations
                    self.markov[state_pripri][state][k] = self.pert_survival[pi] 
                        * sum_piprime; 
                }
            }
        }
    }

    fn Best_Response(&mut self) {
        let mut V_max;
        let mut Vn = vec![0.;G];
        let mut max_k;
        let mut dif_v = 1.;
        while dif_v > self.tolv{
            for state in 0..N_states{
                max_k = self.x[state];
                V_max = 0.;
                for k in 0..G{
                    // Calculate expected fitness for action k for prior state state
                    Vn[k] =0.;
                    for state_prime in 0..N_states{
                        Vn[k] += self.markov[state_prime][state][k]*self.V[0][state_prime];
                    }

                    // Check if new best response for current state
                    if Vn[k] > V_max {
                        max_k = k;
                        V_max = Vn[k];
                    }
                }
                self.V[1][state] = self.r_state(state) + Vn[max_k];
                self.x_prime[state] = max_k;

            }
            dif_v = self.difv();
            self.V[0] = self.V[1].clone();
        }
    }

    fn evo_dif(&self) -> f64 {
        let dif = (0..N_states)
            .map(|state| (self.x[state] as f64 - self.x_prime[state] as f64).abs())
            .sum();
        return dif
    }
    
    fn ploth(&self){
        for h in &self.hi{
            print!("{}",h);
        }
    }

    fn run(&mut self){
        self.inits();


        // evolutionary stability:
        let mut difx: f64 = 1.;
        while difx > 0.{

            // demographic stability:
            self.demog();
            self.update_markovian();

            // plot hi
            self.ploth();

            // find best response strategy
            self.Best_Response();

            // print x
            for pi in 0..LENPI{
                println!("state s:{}, pi:{}, m:{}, best strat x: {}", NS-2, pi, NM/2, self.x[idx(NS-2, pi, NM/2)]);
            }

            // Calculate difference in strategy
            difx = self.evo_dif();

            // update x
            self.x = self.x_prime.clone();
        }
    }

}


// main Functions
fn main() -> std::io::Result<()>  {
// // Comand-line arguments
//     let args: Vec<String> = env::args().collect();
//     let project_id = &args[1];
//     let path = format!("./Results/{}/", project_id);
//     // Construct the full path
//     let path_construct = Path::new(&path);
//     // Ensure parent directory exists
//     let _ = fs::create_dir_all(path_construct); // Create directory path if it doesn't exist
//     let _ = write_csv_header(&path);

/////////////////////////////////////////// Initialise parameters \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    let belief = BeliefState{
        pi0: 0.,
        pi1: 0.,
        pi2: 0.,
    };

    let mut env0 = Environment {
        hi: vec![0.;I], //(i) pdf over patch population densities
        hi_prime: vec![0.;I], //(i) pdf over patch population densities post some event
        n_max: 100., // population bin sizes
        g_mean: 1., // mean food mass
        g_sd: 2., // standard deviation of food mass
        m_switch: 5, //threshold m above which fighter phenotype is expressed
        r_fighter: 2., // fitness benefit of fighter phenotype
        r_m: 10., // fitness benefit of quality
        r_baseline: 10., // baseline fitness
        a_qi: vec![0.;I], //(i) arrivals from dispersal
        mu: 0.1, // Dispersal rate
        r_qi: vec![0.;I], //(i) arrivals from local births
        R: vec![vec![vec![0.;I];NM];NS], //(s,m,i) fecundity of individual of size s and maturity m
        rhoi: vec![vec![0.;I];N_states], //(idx,i) prob. distribution over states for individuals in patches of density qi
        p_births: vec![vec![0.;I];I], //(i,j)
        rhotilde: vec![0.;N_states], //(idx)
        p_g_i: vec![vec![0.;I];G],
        pipig: vec![vec![vec![0.;G];LENPI];LENPI], //Trans. prob. og pi prime given pi and g [piprime][pi][g]
        beliefs: vec![belief;LENPI],
        x: vec![G-1;N_states], // (idx) gives resident strategy -> threshold value above which input energy goes to quality
        dev_m: vec![vec![vec![vec![0.;G];G];N_states];N_states], // (idx prime, idx, g, kappa) developmental transitions
        y_max: G as f64 /2., // maximum maintenance investment
        survival: vec![vec![0.;G];N_states], // ((s,pi,m),g) prob. survival
        lambda: 0.1, // Baseline rate of death
        rhohat: vec![vec![0.;I];N_states], // state distribution after development fatalities, before normalisation
        pfail: vec![vec![0.;I];N_states], // prob of death for current physiological and habitat state over development
        sigma_per: 2., // standard deviation of patch quality perturbations
        p_tau: vec![0.;2*G], // pseudonormal probability distribution over perturbations
        gamma: 0.5, // parameter determining maximum effect of demographic feedback on patch survival
        c: vec![0;I], // (i) rounded shift in prob. of posterior patch state after perturbatins given pop. density i
        tol: 0.001, // tolerance threshold for difference in demographic state, below which projections end
        p_pert: vec![vec![0.;I];I], // transition probabilities of patches given perturbations
        x_prime: vec![G-1;N_states], // Best response strategy
        ptilde: vec![vec![0.;I];I], // (i_prime,i) Evolved prior for transition prob. of patches at demographic stability
        V: vec![vec![0.;2];N_states], // (state,n) Fitness
        tolv: 0.001, // tolerance threshold in the difference in v for finding the best response
        pipitime: vec![vec![0.;LENPI];LENPI], // (piprime,pi) prob of belief transition given time
        p_g_pi: vec![vec![0.;LENPI];G], // (g,pi) the probability of food mass g given prior pi
        pert_survival: vec![0.;LENPI], // (pi) probability of survival given prior belief pi
        markov: vec![vec![vec![0.;G];N_states];N_states], //(state,stateprime,kappa) transition prob given action kappa
    };

    env0.run();

// // start r session
//      let mut r = RSession::new()?;
//     r.exec("source('src/plots.r')")?;
//     let r_mutex = Mutex::new(r);

// // climate stochasticity: 
//     let path = format!("./Results/{}/stoch/", project_id);
//     let path_construct = Path::new(&path);
//     // Ensure parent directory exists
//     let _ = fs::create_dir_all(path_construct); // Create directory path if it doesn't exist
//     let _ = write_csv_header(&path);
//     (0..iterations).into_par_iter().for_each(|g|  {
//         // Initialise stochastic variables
//         let mut rng = rand::thread_rng();
//         let habitat;
//         if rng.gen::<f64>() < 0.5 {
//             habitat = 1;
//         } else {
//             habitat = 0;
//         }
//         let mut env = env0.clone();
//         let mut agent = agent0.clone();
//         agent.mutate(1.0, 1.0); // randomize resident loci
//         env.pop = init_pop(1000, agent);   
//         env.habitat = habitat;
//         let x = rng.gen_range(0.0..1.0); // uniform sample from parameter space
//         env.stoch = x;
        
//         println!("Simulation started: stoch: {}, trial: {}", x, g);
//         run(
//             generations, 
//             &path, 
//             Some(&(x.to_string()+"_stoch_")), 
//             Some(&g),
//             env
//         );
//         println!("Simulation done: stoch: {}, trial: {}", x, g);
//         if g % 5 == 0 {
//             let mut r_guard = r_mutex.lock().unwrap();
//             r_guard.exec(&format!("run_stoch_plot('{}')", path)).unwrap();
//         }
//     });
//     let mut r = RSession::new()?;
//     r.exec(&format!("run_stoch_plot('{}')", path)).unwrap();

    Ok(())
}