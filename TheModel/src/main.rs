#![allow(nonstandard_style)]
extern crate csv;
use std::error::Error;
use std::fs::OpenOptions;
// use std::iter::Sum;
use std::println;
use std::vec;
// use rayon::prelude::*;
use std::env;
use std::fs::{self, File};
use std::path::Path;
use csv::Writer;
use std::io::BufWriter;

// R stuff
use std::process::{Command, Stdio};
use std::io::Write;
// use std::sync::Mutex;

pub struct RSession {
    child: std::process::Child,
}

impl RSession {
    pub fn new() -> std::io::Result<Self> {
        let child = Command::new("R")
            .args(["--vanilla", "--slave"])
            .stdin(Stdio::piped())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
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

    let file = match File::create(&path) {
        Ok(f) => f,
        Err(e) => return Err(Box::new(e)),
    };

    let buf_writer = BufWriter::new(file);
    let mut wtr = Writer::from_writer(buf_writer);

    wtr.write_record(&[
        "n_min",
        "g_mean",
        "g_sd",
        "f_switch",
        "r_fighter",
        "r_e",
        "r_baseline",
        "mu",
        "c_max",
        "lambda",
        "gammah",
        "gammal",
        "s",
        "i", 
        "e", 
        "xstar",
        "rho_0",
        "rho_1",
        "eta",
        "P00",
        "P10",
        "P01",
        "P11"
    ])?;

    wtr.flush()?;
    Ok(())
}

fn write_csv_headerLH(path: &str) -> Result<(), Box<dyn std::error::Error>> {

    let file = match File::create(&path) {
        Ok(f) => f,
        Err(e) => return Err(Box::new(e)),
    };

    let buf_writer = BufWriter::new(file);
    let mut wtr = Writer::from_writer(buf_writer);

    wtr.write_record(&[
        "t", "s", "i", "e", "n", "rho", "x"
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
const S: usize = 11;
const I: usize = 7;
const E: usize = 10;
const N: usize = 2;
const G: usize = 5;
const Kappa: usize = 5;
const N_states: usize = S * I * E;

#[derive(Clone, Debug)]
struct Environment {
    etatilde: f64, // prior for the patch density distribution
    n_min: f64,
    g_mean: f64,
    g_sd: f64,
    f_switch: f64, //threshold investment above which fighter phenotype is expressed at maturity
    r_fighter: f64,
    r_e: f64, // benefit of reserves
    r_baseline: f64,
    lambda: f64, // Dispersal rate
    mu_e: f64, // mortality rate due to low energy reserves
    mu_base: f64, // baseline mortality rate
    c_max:f64, // maximum maintenance investment
    gammah: f64, // parameter transition prob from high to low
    gammal: f64, // parameter transition prob from low to high
    eta: f64, // proportion of patches that are high density
    eta_prime: f64, //  proportion of patches that are high density, post some event
    u_q: Vec<f64>, //(i) survivals from development
    a_q: Vec<f64>, //(i) arrivals from dispersal
    r_q: Vec<f64>, //(i) arrivals from local births
    R: Vec<Vec<f64>>, //(s,i,e,kappa) fecundity of individual of state (s,i,e) given investment kappa
    rhon: Vec<Vec<f64>>, //(idx,i) prob. distribution over states for individuals in patches of density qi
    rhotilde: Vec<f64>, //(idx) distribution of newborns
    L_g_n: Vec<Vec<f64>>,
    pipig: Vec<Vec<Vec<f64>>>, //Trans. prob. og pi prime given pi and g [piprime][pi][g]
    x: Vec<usize>, // (idx) gives resident strategy -> threshold value above which input energy goes to quality
    dev_m: Vec<Vec<Vec<f64>>>, // (idx prime, "survives"| idx, kappa) developmental transitions
    M: Vec<Vec<Vec<f64>>>, // (idx prime, "survives"| idx, kappa) state transitions
    rhohat: Vec<Vec<f64>>, // state distribution after development fatalities, before normalisation
    tol: f64, // tolerance threshold for difference in demographic state, below which projections end
    p_stoch: Vec<Vec<f64>>, // transition probabilities of patches given demo. stoch. & perturbations
    x_prime: Vec<usize>, // Best response strategy
    ptilde: Vec<Vec<f64>>, // (i_prime,i) Evolved prior for transition prob. of patches at demographic stability
    pno: Vec<Vec<f64>>, // (i_prime,i) latent transition prob. of patches between cycles
    pdet: Vec<Vec<f64>>, // (i, i_prime) latent transition prob. of patches between cycles given deterministic processes
    V: Vec<Vec<f64>>, // (n,state) Fitness
    tolv: f64, // tolerance threshold in the difference in v for finding the best response
    pipitime: Vec<Vec<f64>>, // (piprime,pi) prob of belief transition given time
    p_g_i: Vec<Vec<f64>>, // (g,pi) the probability of food mass g given prior pi
    pert_survival: Vec<f64>, // (pi) probability of survival given prior belief pi
    pcsurvival: Vec<f64>, // survival given state e
    pceeg: Vec<Vec<Vec<f64>>>, // transitions post foraging and maintenance
}

// Implementations
fn idx(s:usize, i:usize, e:usize) -> usize {
    // returns index integer from individual state
    s * I * E + i * E + e
}

impl Environment { 
    fn q(&self, n: usize)->f64{
        if n == 0 {
            return self.n_min;
        }else{
            return self.n_min*5.;
        }
    }

    fn interpolate_n(&self, mut prime:f64) -> (usize,usize,f64,f64) {
        // Clamp the posterior population size
        prime = prime.min(self.q(1)).max(self.q(0));
        
        prime = (prime - self.q(0))/(self.q(1)-self.q(0));

        return (0,1,1.-prime,prime)
    }

    fn init_R(&mut self){
        let mut r:f64;
        for e in 0..E{
            for i in 0..I{
                for s in 0..S{
                    for kappa in 0..Kappa{
                        r = 0.;
                        
                        if s == S-2 {
                            r = (kappa as f64)/(Kappa as f64 - 1.) * (e as f64) * (self.r_e);
                        } else if s == S-1 {
                            r = (kappa as f64)/(Kappa as f64 - 1.) * (e as f64) * (self.r_e + self.r_fighter);
                        }

                        self.R[idx(s,i,e)][kappa] = r;
                    }
                }
            }
        }
    }

    fn update_r(&mut self){
        // Gives the rate of reproduction for each patch density given 
        // the distribution of physiological states and the resident strategy
        for n in 0..N {
            self.r_q[n] = 0.0;

            for state in 0..N_states{
                self.r_q[n] += self.rhon[state][n]*self.R[state][self.x[state]];
            }
        }
    }

    fn interpolate_pn(&self, pn: f64) -> (usize,usize,f64,f64){
        
        // Interpolation step: decompose into interger and fractional parts
        let scale = (I - 1) as f64;
        let i1 = (scale * pn).floor();
        let i2 = (i1+1.).min(I as f64-1.);

        let p2 = scale * pn - i1;
        let p1 = 1. - p2;

        return (i1 as usize, i2 as usize, p1, p2)
    }

    fn newborns(&mut self){
        self.rhotilde.fill(0.0);

        // interpolating priors onto the grid
        let (i0_prior, i1_prior, p0, p1) = self.interpolate_pn(self.etatilde);

        // Add to starting line
        self.rhotilde[idx(0,i0_prior,1)] += p0;
        self.rhotilde[idx(0,i1_prior,1)] += p1;
        // if p0+p1+p2 > 1. {
        //     println!("whaaaaaaaaaaaaaaaaaaaaaaaaattf, priors: {},{},{}, distr: {},{},{}", i0_prior, pi1_prior, pi2_prior, p0,p1,p2);
        // }
    }

    fn init_L_g_n(&mut self){
        // This function updates the probability of a food mass of g in a patch of density n
        // as per the first equaitions of section 3.3 defining the pdf over food masses.
        let mut q_n:f64;
        for n in 0..N{
            q_n = self.q(n);
            // Equation 18
            let K: f64 = (0..G)
                .map(|g| (-((g as f64-self.g_mean*self.q(0)/q_n).powf(2.))/(2.*self.g_sd.powf(2.))).exp())
                .sum();

            // Equation 17
            for g in 0..G{
                self.L_g_n[g][n] = (1./K) * (-((g as f64-self.g_mean*self.q(0)/q_n).powf(2.))/(2.*self.g_sd.powf(2.))).exp();
                println!("g: {}, n: {}, L(g|n): {}", g, n, self.L_g_n[g][n]);
            }
        }
    }

    fn init_pipig(&mut self){
        // This function defines the transition probabilities of beliefs
        // for each given food mass of g
        let mut pi1;
        let mut prime1;
        for vec in &mut self.pipig {
            for vec2 in vec {
                vec2.fill(0.);
            }
        }
        for pn in 0..I{
            for g in 0..G{
                // extract belief distribtion from flat pi vertex index
                pi1 = pn as f64/(I as f64 -1.);

                // Find posterior belief distribution
                prime1 = (pi1 * self.L_g_n[g][1])/((1.-pi1) * self.L_g_n[g][0] + pi1 * self.L_g_n[g][1]);
                // println!("prim1: {}, prim2: {}, prim3: {}, total: {}, pis: {},{},{}", prime1, prime2, prime3, prime1+prime2+prime3, pi1, pi2, pi3);

                // Interpolate
                let (n1, n2, p1, p2) = self.interpolate_pn(prime1);
                
                // add
                self.pipig[n1][pn][g] += p1;
                self.pipig[n2][pn][g] += p2;
                   
            }
        }
    }

    fn c(&self, s:f64, e:f64, g:f64) -> f64{
        let size3 = (s+e).powf(3.);
        return e + g - size3*self.c_max
    }

    fn init_maintenance(&mut self){
        for state in 0..N_states{
            
            let (s, _i, e) = self.state_invert(state);
            
            // Define probability of survival given energy reserves
            self.pcsurvival[state] = (1. - (E as f64 - 1. - e as f64)/(E as f64 - 1.) * self.mu_e) * (1. - self.mu_base);

            // Define transition probabilities of energy reserves given maintenance costs
            for g in 0..G{
                
                let epost = self.c(s as f64, e as f64, g as f64).min(E as f64 -1.).max(0.);
                let (e1, p1, e2, p2) = self.interpolate_e(epost);
                self.pceeg[e1][state][g] += p1;
                self.pceeg[e2][state][g] += p2;
            }
        }
    }

    fn interpolate_e(&self, mut post_e:f64) -> (usize,f64,usize,f64){
        post_e = post_e.min((E-1) as f64);
        let e1 = (post_e).floor();
        let p2 = post_e-e1;
        let e2 = (e1+1.).min((E-1) as f64);
        let p1 = 1.-p2;

        return (e1 as usize, p1, e2 as usize, p2)
    }

    fn interpolate_s(&self, mut post_s:f64) -> (usize,f64,usize,f64){
        post_s = post_s.min((S-1) as f64);
        let s1 = (post_s).floor();
        let p2 = post_s-s1;
        let s2 = (s1+1.).min((S-1) as f64);
        let p1 = 1.-p2;

        return (s1 as usize, p1, s2 as usize, p2)
    }

    fn state_invert(&self, state:usize) -> (usize,usize,usize){
        let e = state % E;
        let i = (state / E) % I;
        let s = state / (I * E);

        return (s, i, e)
    }

    fn init_pstoch(&mut self) {        
        // transition probabilities given demographic stability and density dependent degradation, section 3.4
        self.p_stoch[0][0] = 1. - self.gammal;
        self.p_stoch[1][0] = self.gammal;
        self.p_stoch[0][1] = self.gammah;
        self.p_stoch[1][1] = 1. - self.gammah;
        println!("pstoch 0|0: {}, 1|0: {}, 0|1: {}, 1|1: {}", 
            self.p_stoch[0][0], 
            self.p_stoch[1][0],
            self.p_stoch[0][1],
            self.p_stoch[1][1]);
    }

    fn update_pipitime(&mut self){
        // what is the probability of posterior belief i prime 
        // given prior i and the passing of time

        for pi in &mut self.pipitime{
            pi.fill(0.);
        }

        for i in 0..I{
            
            let pn1 = i as f64/(I as f64 - 1.);
            let pn0 = 1. - pn1;

            let post = pn0*self.ptilde[1][0] +pn1*self.ptilde[1][1];

            let (d1, d2, p1, p2) = self.interpolate_pn(post); 
            self.pipitime[d1][i] += p1;
            self.pipitime[d2][i] += p2;
        }
    }

    fn Dem_Dif(&self, oldh:f64) -> f64 {
        let dif = (&self.eta - oldh).abs();
        return dif
    }

    fn init_rho(&mut self){
        for n in 0..N{
            for state in 0..N_states{
                self.rhon[state][n] = self.rhotilde[state];
            }
        }
    }

    fn food_sum(&mut self){
        for state in 0..N_states{
            for state_prime
        }
        self.F[idx(s,iprime, eprime)]
    }

    fn inits(&mut self){
        println!("Initialising priors");
        self.eta = 0.5;
        self.etatilde = 0.5;
        self.init_pstoch();
        self.ptilde = self.p_stoch.clone();

        println!("Initialising rho");
        self.newborns();
        self.init_rho();

        // as per section 3.3 in finding demographic stability.
        println!("Initialising L(g|q_n)");
        // first update food mass pdf:
        self.init_L_g_n();

        // Initialise the transition probabilities from pi to pi^prime given a food mass of g
        println!("Initialising P(pi|pi,g)");
        self.init_pipig();

        for g in 0..G{
            for pi in 0..I{
                let mut sum = 0.; 
                for piprime in 0..I{
                    sum +=  self.pipig[piprime][pi][g];
                }
                println!("pi: {}, g: {}, sumpipig: {}", pi as f64/I as f64, g, sum);
            }
        }

        // init reproductive rates
        println!("Initialising R");
        self.init_R();

        // init prob of g given prior pi
        println!("Initialising P(g|pi)");
        self.init_pgi();

        // Init prob of survival given maintenance and transition probabilities of energy reserves given maintenance costs
        self.init_maintenance();
        
        // init prob of survival given perturbations
        println!("Initialising P(survival|pert");
        self.init_pertsurvival();

        // init prob of developmental transitions
        println!("Initialising P(state_prime|state,kappa,g)");
        self.init_dev_m_kappa();

        // update belief update given time
        println!("Initialising P(piprime|pi,time)");
        self.update_pipitime();
            
        let rhosum0:f64  = (0..N_states)
            .map(|state| self.rhon[state][0])
            .sum();
        let rhosum1:f64 = (0..N_states)
            .map(|state| self.rhon[state][1])
            .sum();
        
        println!("initial rhoi sums: {}, {}", rhosum0, rhosum1);
        println!("initial priors for P(n|o) values: P(0|0) = {}, P(0|1) = {}, P(1|0) = {}, P(1|1) = {}", self.ptilde[0][0],  self.ptilde[0][1], self.ptilde[1][0], self.ptilde[1][1]);
        println!("initial eta: {}", self.eta);
        
        println!("Initialisations done!");
            // let mut l0 = 0.;
            // let mut l1 = 0.;
            // for g in 0..G{
            //     l0 += self.L_g_i[g][0];
            //     l1 += self.L_g_i[g][1];
            // }
            // println!("L0: {}, L1:{}", l0, l1);
    }

    fn init_dev_m_kappa(&mut self){
        // Udates the state transition matrix, dev_M, given all possible actions
        
        // Set developmental transition matrix to equal 0 for all prior and posterior states
        for prior in self.dev_m.iter_mut() {
            for post in prior.iter_mut() {
                post.fill(0.); // for each prior and post states (s and s'), over all g: D(s'|s,g)=0
            }
        }

        let mut investment: f64;
        // for each prior state
        let (mut s, mut pi, mut e):(usize,usize,usize);
        for state in 0..N_states {
            (s,pi,e) = self.state_invert(state);
            for k in 0..G{
                // After development:
                investment = (e as f64 * (k as f64)/(G as f64 - 1.)).max(0.);

                // Calculate the change in state expected for the investments del_m and del_s
                let del_epost = e as f64 - investment;
                let del_spost = investment/(E as f64-1.);
                // Interpolate onto s and m grids:
                let (e1,ep1,e2,ep2) = self.interpolate_e(del_epost);
                let (s1,sp1,s2,sp2) = self.interpolate_s(del_spost+(s as f64));

                // Developmental state transitions
                if s >= S-2{ // development of offspring at maturity (s stays the same)
                    self.dev_m[idx(s, pi, e1)][idx(s, pi, e)][k] = ep1;
                    self.dev_m[idx(s, pi, e2)][idx(s, pi, e)][k] = ep2;
                } else if s == S-3 { // develop into one of two possible morphs
                    if del_spost > self.f_switch{ // Definitely matures, with some prob of fighter
                        let pfighter = (del_spost - self.f_switch)/(1. - self.f_switch);
                        let pscrambler = 1. - pfighter;
                        self.dev_m[idx(S-1, pi, e1)][idx(s, pi, e)][k]+=ep1*pfighter;
                        self.dev_m[idx(S-2, pi, e1)][idx(s, pi, e)][k]+=ep1*pscrambler;
                        self.dev_m[idx(S-1, pi, e2)][idx(s, pi, e)][k]+=ep2*pfighter;
                        self.dev_m[idx(S-2, pi, e2)][idx(s, pi, e)][k]+=ep2*pscrambler;
                    } else { // possibly matures into a scrambler
                        let pscrambler = del_spost / self.f_switch;
                        let psminus3 = 1. - pscrambler;
                        self.dev_m[idx(S-3, pi, e1)][idx(s, pi, e)][k]+=ep1*psminus3;
                        self.dev_m[idx(S-2, pi, e1)][idx(s, pi, e)][k]+=ep1*pscrambler;
                        self.dev_m[idx(S-3, pi, e2)][idx(s, pi, e)][k]+=ep2*psminus3;
                        self.dev_m[idx(S-2, pi, e2)][idx(s, pi, e)][k]+=ep2*pscrambler;

                    }
                } else { // development of life stage/morph at juvenile stages
                    self.dev_m[idx(s1, pi, e1)][idx(s, pi, e)][k]+=ep1*sp1;
                    self.dev_m[idx(s2, pi, e1)][idx(s, pi, e)][k]+=ep1*sp2;
                    self.dev_m[idx(s1, pi, e2)][idx(s, pi, e)][k]+=ep2*sp1;
                    self.dev_m[idx(s2, pi, e2)][idx(s, pi, e)][k]+=ep2*sp2;

                    // println!("post {}, {}, {}, {}, mu: {}", mp1*sp1*mu, mp1*sp2*mu, mp2*sp1*mu, mp2*sp2*mu, mu);
                }
                
            }
        }
    }

    fn Development(&mut self){
        // This function will do all the updates associated with strategic development;
        // including the development of offspring;
        // and including updating of beliefs given evolve priors for the transition probabilities between patches.

        // Development of offspring!
        // first we update r_q which will later give the number of offspring developed
        self.update_r();
        // Update newborn physiology
        self.newborns();

        // Development between life stages!
        for n in 0..N{
            // set post to zero
            self.rhohat[n].fill(0.);
            for state_prime in 0..N_states{
                let (sprime, iprime, eprime) = self.state_invert(state_prime);
                // demographic projection
                for state in 0..N_states{
                    let (_s, i, _e) = self.state_invert(state);
                    self.rhohat[n][state_prime] += self.pipitime[iprime][i] * self.dev_m[idx(sprime, i, eprime)][state][self.x[state]] * self.rhon[state][n];
                }
            }
        }       
        self.rhon = self.rhohat.clone(); 
    }

    fn Foraging(&mut self){
        // Individuals forage food and pay their maintenance costs, with some probability
        for n in 0..N{
            for stateprime in 0..N_states{
                self.rhohat[n][stateprime] = 0.;
                let (_sprime, iprime, eprime) = self.state_invert(stateprime);
                
                for i in 0..I{
                    for e in 0..E{
                        for g in 0..G{
                            self.rhohat[n][stateprime] += self.rhon[state][n] *
                                self.pcsurvival[stateprime] * 
                                self.pceeg[eprime][e][g] *
                                self.pipig[iprime][i][g] *
                                self.L_g_n[g][n];
                        }
                    }
                }
            }
            // proportion that survived in patches of density n
            self.u_q[n] = (0..N_states)
                .map(|state| self.q(n)*self.pcsurvival[state]*self.rhohat[n][state])
                .sum();

            // Update rhoi
            let sumhat:f64 = (0..N_states)
                .map(|state| self.rhohat[n][state]).sum();
            println!("sumhat:{}",sumhat);
            for state in 0..N_states{
                self.rhon[state][n] = self.rhohat[n][state] / sumhat;
            }
        }
    }

    fn Growth(&mut self){
        // This function projects the distributions h_qi and rho given dispersal of newborns and deaths

        // Clean pdet
        self.pdet[0].fill(0.0);
        self.pdet[1].fill(0.0);

        // Set rho prime to zero so we can add to it.
        let mut rhonprime: Vec<Vec<f64>> = vec![vec![0.0; N]; N_states];

        // 3.2.2 Patch projection
        // Second expression of equation 4 defining alpha, representing the number of migrants
        let dispersing_pool: f64 = self.eta*self.q(1)*self.r_q[1] + (1.-self.eta)*self.q(0)*self.r_q[0];

        for n in 0..N {

            // calculating alpha, equation 4
            self.a_q[n] = (1. - self.lambda)*self.q(n)*self.r_q[n] + self.lambda*dispersing_pool;

            // calculating the new patch density size after dispersal, equation 6
            let qnprime: f64 = self.a_q[n] + self.u_q[n];

            // Interpolating iprime to find transition probabilities on 1-simplex qi grid, equations 7-10
            let (d1, d2, p1, p2) = self.interpolate_n(qnprime);
             
            self.pdet[d1][n] += p1;
            self.pdet[d2][n] += p2;
        }

        println!("u0: {}, a0: {}, q0:{}",self.u_q[0], self.a_q[0], self.q(0));
        println!("u1: {}, a1: {}, q1:{}",self.u_q[1], self.a_q[1], self.q(1));
        println!("p_det (0|1): {}, (0|0): {}, (1|0): {}, (1|1): {}", self.pdet[0][1], self.pdet[0][0], self.pdet[1][0], self.pdet[1][1]);

        // define pno
        self.pno[0][1] = self.p_stoch[0][1]*self.pdet[1][1] + self.p_stoch[0][0]*self.pdet[0][1];
        self.pno[0][0] = self.p_stoch[0][1]*self.pdet[1][0] + self.p_stoch[0][0]*self.pdet[0][0];
        self.pno[1][0] = self.p_stoch[1][1]*self.pdet[1][0] + self.p_stoch[1][0]*self.pdet[0][0];
        self.pno[1][1] = self.p_stoch[1][1]*self.pdet[1][1] + self.p_stoch[1][0]*self.pdet[0][1];
        // self.pno[1][1] = 1.;
        // self.pno[0][0] = 1.;
        println!("p(0|1): {}, p(0|0): {}, p(1|0): {}, p(1|1): {}", self.pno[0][1], self.pno[0][0], self.pno[1][0], self.pno[1][1]);
        // Patch state distribution projections
        self.eta_prime = self.eta * self.pno[1][1] + (1.-self.eta) * self.pno[1][0];
            
        // Now we move on to projection of physiological states:
        for state in 0..N_states {
            rhonprime[state][1] += (1./self.eta_prime) * self.eta*self.pno[1][1]*((self.u_q[1]/(self.a_q[1] + self.u_q[1]))*self.rhon[state][1] + (self.a_q[1]/(self.a_q[1] + self.u_q[1]))*self.rhotilde[state]);
            rhonprime[state][1] += (1./self.eta_prime) * (1.-self.eta)*self.pno[1][0]*((self.u_q[0]/(self.a_q[0] + self.u_q[0]))*self.rhon[state][0] + (self.a_q[0]/(self.a_q[0] + self.u_q[0]))*self.rhotilde[state]);
            rhonprime[state][0] += (1./(1.-self.eta_prime)) * self.eta*self.pno[0][1]*((self.u_q[1]/(self.a_q[1] + self.u_q[1]))*self.rhon[state][1] + (self.a_q[1]/(self.a_q[1] + self.u_q[1]))*self.rhotilde[state]);
            rhonprime[state][0] += (1./(1.-self.eta_prime)) * (1.-self.eta)*self.pno[0][0]*((self.u_q[0]/(self.a_q[0] + self.u_q[0]))*self.rhon[state][0] + (self.a_q[0]/(self.a_q[0] + self.u_q[0]))*self.rhotilde[state]);
                    
            if self.eta_prime == 0. {
                rhonprime[state][1] = self.rhotilde[state];
            }
            if 1. - self.eta_prime == 0. {
                rhonprime[state][0] = self.rhotilde[state];
            }
            if rhonprime[state][0].is_nan(){
                println!("eta: {}", self.eta);
                println!("u0: {}, a0: {}, q0:{}",self.u_q[0], self.a_q[0], self.q(0));
                println!("pno11:{}, pno10:{}",self.pno[1][1], self.pno[1][0]);
                println!("n: {}, etaprime: {}", 0, self.eta_prime);
                std::process::exit(0);
            }
            if rhonprime[state][1].is_nan(){
                println!("eta: {}", self.eta);
                println!("ui: {}, ai: {}, qi:{}",self.u_q[1], self.a_q[1], self.q(1));
                println!("pno11:{}, pno10:{}",self.pno[1][1], self.pno[1][0]);
                println!("n: {}, etaprime: {}", 1, self.eta_prime);
                std::process::exit(0);
            }
        }

        // print some shit
        let sumthin:f64 = (0..N_states)
            .map(|state| (self.q(0)/self.q(1))*self.rhon[state][0] + (1.-self.q(0)/self.q(1))*self.rhotilde[state])
            .sum();
        let rhotildesum:f64=(0..N_states)
            .map(|state| self.rhotilde[state])
            .sum();
        println!("sumsumsum: {}, qsums: {}, rhotildesum: {}", sumthin, (self.q(0)/self.q(1))+(1.-self.q(0)/self.q(1)), rhotildesum);

        
        // make rhoi rhoiprime
        self.rhon = rhonprime.clone();

        // update eta
        self.eta = self.eta_prime;
    }

    fn demog(&mut self){
        // demographic stability:
        let mut dif_h:f64 = 1.;
        let mut oldh:f64 = self.eta;
        let mut cycle = 0;

        // projection algorithm, i.e. the growth cycle
        while dif_h > self.tol{ 
            cycle+=1;
            println!("\nGrowth Cycle {}", cycle);
            self.Development();
            println!("post development eta: {}", self.eta);
            if self.eta.is_nan(){
                    std::process::exit(0);
            }
            
            let rhosum0:f64  = (0..N_states)
                .map(|state| self.rhon[state][0])
                .sum();
            let rhosum1:f64 = (0..N_states)
                .map(|state| self.rhon[state][1])
                .sum();
            println!("rhon sums: {}, {}", rhosum0, rhosum1);

            self.Foraging();
            println!("post foraging eta: {}", self.eta);
            if self.eta.is_nan(){
                std::process::exit(0);
            }
            
            let rhosum0:f64  = (0..N_states)
                .map(|state| self.rhon[state][0])
                .sum();
            let rhosum1:f64 = (0..N_states)
                .map(|state| self.rhon[state][1])
                .sum();
            println!("rhon sums: {}, {}", rhosum0, rhosum1);
            
            self.Growth();
            println!("post births and deaths (i.e. growth): {}", self.eta);
            if self.eta.is_nan(){
                std::process::exit(0);
            }
            
            let rhosum0:f64  = (0..N_states)
                .map(|state| self.rhon[state][0])
                .sum();
            let rhosum1:f64 = (0..N_states)
                .map(|state| self.rhon[state][1])
                .sum();
            println!("rhon sums: {}, {}", rhosum0, rhosum1);


            dif_h = self.Dem_Dif(oldh);
            oldh = self.eta;

            println!("dif_h: {}, eta: {}", dif_h, self.eta);
            if self.eta.is_nan(){
                std::process::exit(0);
            }
        }
        println!("Demog done: eta: {}, pno00:{}, pno10:{}, pno01:{}, pno11:{}", self.eta, self.pno[0][0], self.pno[1][0], self.pno[0][1], self.pno[1][1]);


    }

    fn difv(&self) -> f64 {
        let tot:f64 = (0..N_states)
            .map(|state| (self.V[0][state] - self.V[1][state]).abs())
            .sum();
        return tot
    }

    fn init_pgi(&mut self){
        for pi in 0..I{
            for g in 0..G{
                let p1 = pi as f64/(I as f64 -1.);
                let p0 = 1. - p1;
                self.p_g_i[g][pi] = self.L_g_n[g][0]*p0 + self.L_g_n[g][1]*p1;

                println!("P: {}, g: {}, pi: {}",self.p_g_i[g][pi], g, p1);
            }
        }
    }

    fn init_pertsurvival(&mut self){
        // calcilate probability of survival for an individual of belief state pi, given transitions in patch population densitites and random deaths
        for pi in 0..I{
            let prior = pi as f64/(I as f64 -1.);
            
            self.pert_survival[pi] = (1. - prior) + prior * (self.p_stoch[1][1] + self.p_stoch[0][1] * (self.q(0)/self.q(1)));
            
        }
    }
    
    fn update_M(&mut self){
        let (mut sprime, mut ipripri, mut epripri);
        let (mut _s, mut i, mut _e);
        for stateprime in 0..N_states{
            (sprime, ipripri, epripri) = self.state_invert(stateprime);
            for state in 0..N_states{
                (_s, i, _e) = self.state_invert(state);
                for kappa in 0..Kappa{
                    self.M[stateprime][state][kappa] = 0.;
                    for iprime in 0..I {
                        for g in 0..G {
                            for eprime in 0..E{
                                self.M[stateprime][state][kappa] += 
                                    self.pcsurvival[stateprime] *
                                    self.pipitime[ipripri][iprime] * 
                                    self.pert_survival[iprime] *
                                    self.p_g_i[g][i] *
                                    self.pceeg[epripri][eprime][g] *
                                    self.pipig[iprime][i][g] *
                                    self.dev_m[idx(sprime, i, eprime)][state][kappa];
                            }
                        }
                    }
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
                // find best action kappa given prior states s and m, posterior piprime, and food mass g
                for k in 0..G{
                    // k given prior pi^primeprime
                    Vn[k]=0.;
                    for state_prime in 0..N_states{
                        // value of action kappa
                        Vn[k] += self.R[state][k] + self.M[state_prime][state][k] * self.V[1][state_prime];                                
                    }
                            
                    // Check if new best response for current state
                    if Vn[k] >= V_max {
                        max_k = k;
                        V_max = Vn[k];
                    }
                }

                // running best response
                self.x_prime[state] = max_k;

                // Weight posterior V by probability of posterior piprime, probability of survival for pi, and probability of g given pi
                self.V[0][state] = V_max;
            }

            dif_v = self.difv();

            self.V[1] = self.V[0].clone();
            println!("V updated, current difference: {}", dif_v);
        }

        println!("V converged on V*!");
    }

    fn evo_dif(&self) -> f64 {
        let dif = (0..N_states)
            .map(|state| (self.x[state] as f64 - self.x_prime[state] as f64).abs())
            .sum();
        return dif
    }
    
    fn run(&mut self, file:&str){
        self.inits();


        // evolutionary stability:
        let mut difx: f64 = 10.;
        while difx > 0.{

            // demographic stability:
            println!("\nRunning demographic projection algorithm...");
            self.demog();

            // evolving priors
            println!("\nEvolving priors...");
            self.etatilde = self.eta;
            self.ptilde = self.pno.clone();
            self.newborns();
            self.update_pipitime();
            println!("Priors updated!");

            println!("Updating the action and demography dependent Markov state transition matrix");
            self.update_M();

            // find best response strategy
            println!("\nRunning Bellman equation for adapting the developmental strategy...");
            self.Best_Response();
            println!("Therefore: Best response strategy found!");
        
            // Calculate difference in strategy
            println!("\nChecking if the resident strategy was invadable...");
            difx = self.evo_dif();
            println!("The difference in x was: {}", difx);
            if difx > 0.{
                println!("Resident strategy was invaded!");
            }
            // for state in 0..N_states {
            //     println!(
            //         "{}: {} vs {}",
            //         state,
            //         self.x[state],
            //         self.x_prime[state]
            //     );
            // }

            // update x
            self.x = self.x_prime.clone();
        }
    
        println!("Resident strategy was uninvadable, ESS found!");
        // name of file
        for state in 0..N_states{
            let (s, pi, e) = self.state_invert(state);
            // println!("state s:{}, pi:{}, m:{}, ESS x(s,pi,m): {}", s, pi as f64/(NPI as f64 - 1.), m, self.x[state]);
            let v: Vec<f64> = vec![
                self.n_min,
                self.g_mean,
                self.g_sd,
                self.f_switch,
                self.r_fighter,
                self.r_e,
                self.r_baseline,
                self.mu_e,
                self.c_max,
                self.lambda,
                self.gammah,
                self.gammal, 
                s as f64, 
                pi as f64/(I as f64 - 1.), 
                e as f64, 
                self.x[state] as f64, 
                self.rhon[state][0], 
                self.rhon[state][1], 
                self.eta,
                self.pno[0][0],
                self.pno[1][0],
                self.pno[0][1],
                self.pno[1][1]
                ];
            let _=try_print(v, &file);
        }
        
        println!("ESS found and printed to file: {}", &file);
    }

    fn lifehistory(&mut self, file:&str, plab:Vec<Vec<f64>>){
        // init distribution
        for state in 0..N_states{
            self.rhon[state][0] = self.rhotilde[state]*(1.-self.eta)*100.*self.q(0);
            self.rhon[state][1] = self.rhotilde[state]*(self.eta)*100.*self.q(1);
        }
        let mut popsize: f64 = (1.-self.eta)*100.*self.q(0) + (self.eta)*100.*self.q(1);
        let mut t: f64=1.;

        // project distribution
        while popsize > 1.{
            println!("Population size: {}", popsize);

            // print distribution
            for state in 0..N_states{
                let (s,pi,m) = self.state_invert(state);
                for n in 0..N{
                    let v: Vec<f64> = vec![t, s as f64, (pi as f64)/(I as f64 - 1.), m as f64, n as f64, self.rhon[state][n], self.x[state] as f64];
                    let _=try_print(v, &file);
                }
            }


            // calculate posterior distribution to developmnent
            for stateprime in 0..N_states{
                self.rhohat[0][stateprime] = 0.;
                self.rhohat[1][stateprime] = 0.;
                for state in 0..N_states{
                    self.rhohat[0][stateprime] += 
                        self.dev_m[stateprime][state][self.x[state]] *
                        self.rhon[state][0];
                    self.rhohat[1][stateprime] += 
                        self.dev_m[stateprime][state][self.x[state]] * 
                        self.rhon[state][1];
                }
            }

            let mut cumul = 0.;
            for state in 0..N_states{
                cumul += self.rhohat[0][state] + self.rhohat[1][state];
            }
            println!("popsize post development: {}", cumul);
            self.rhon = self.rhohat.clone();

            // calculate posterior to foraging
            for stateprime in 0..N_states{ 
                let (sprime, iprime, eprime) = self.state_invert(stateprime);
                
                for i in 0..I{
                    for e in 0..E{
                        for g in 0..G{
                            self.rhohat[0][stateprime] += self.rhon[idx(sprime,i,e)][0] *
                                self.pceeg[eprime][e][g] *
                                self.pipig[iprime][i][g] *
                                self.L_g_n[g][0];
                            self.rhohat[1][stateprime] += self.rhon[idx(sprime,i,e)][1] *
                                self.pceeg[eprime][e][g] *
                                self.pipig[iprime][i][g] *
                                self.L_g_n[g][1];
                        }
                    }
                }
            }

            // calculate posterior to deaths
            for state in 0..N_states{
                self.rhon[state].fill(0.);
                self.rhon[state][0] += self.pcsurvival[state]*(plab[0][1] * self.rhohat[1][state] * (self.q(0)/self.q(1)) + plab[0][0] * self.rhohat[0][state]);
                self.rhon[state][1] += self.pcsurvival[state]*(plab[1][1] * self.rhohat[1][state] + plab[1][0] * self.rhohat[0][state]);
            }         
            let mut cumul = 0.;
            for state in 0..N_states{
                cumul += self.rhon[state][0] + self.rhon[state][1];
            }
            println!("popsize post plab: {}", cumul);

            // calculate posterior to bayesian updating
            let mut rhoi_prime: Vec<Vec<f64>> = vec![vec![0.0;N]; N_states];
            for state in 0..N_states{
                for n in 0..N{
                    // retrieve state
                    let (s,piprime,e) = self.state_invert(state);

                    // update rhoi
                    rhoi_prime[state][n] = (0..I)
                        .map(|pi| self.pipitime[piprime][pi] * self.rhon[idx(s,pi,e)][n]).sum();
                    
                }
            }
            self.rhon = rhoi_prime;
            let mut cumul = 0.;
            for state in 0..N_states{
                cumul += self.rhon[state][0] + self.rhon[state][1];
            }
            println!("popsize post bayes: {}", cumul);

            // update dif
            popsize = 0.;
            for state in 0..N_states{
                popsize += self.rhon[state][0] + self.rhon[state][1];
            }
            t+=1.;
        }
    }

}

// main Functions
fn main() -> std::io::Result<()>  {
// Comand-line arguments
    let args: Vec<String> = env::args().collect();
    let project_id = &args[1];

    
    // prepare folder for project and ESS results
    let projectpath = format!("./Results/{}/", project_id);
    let path_construct1 = Path::new(&projectpath);
    let _ = fs::create_dir_all(path_construct1); // Create directory path if it doesn't exist
    
    // prepare folder for natural patch dynamics
    let path2 = format!("./Results/{}/natural/", project_id);
    let path_construct2 = Path::new(&path2);
    let _ = fs::create_dir_all(path_construct2); // Create directory path if it doesn't exist

    // prepare folder for lab patch dynamics
    let path3 = format!("./Results/{}/lab/", project_id);
    let path_construct3 = Path::new(&path3);
    let _ = fs::create_dir_all(path_construct3); // Create directory path if it doesn't exist
    
    let policypath = format!("{}Policies.csv", projectpath);
    let _ = write_csv_header(&policypath);
    
    let LHpath1 = format!("{}LifeHistory.csv", path2);
    let _ = write_csv_headerLH(&LHpath1); 
    
    let LHpath2 = format!("{}LifeHistory.csv", path3);
    let _ = write_csv_headerLH(&LHpath2);

/////////////////////////////////////////// Initialise parameters \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    let mut env0 = Environment {
        etatilde: 0.5, // prior for eta
        eta: 0.5, //p(n=1) pdf over patch population densities
        eta_prime: 0.5, //p(n=1) pdf over patch population densities post some event
        n_min: 100., // population bin sizes
        g_mean: (G as f64 -1.), // mean food mass
        g_sd: 1., // standard deviation of food mass
        f_switch: 0.5, // threshold m above which fighter phenotype is expressed
        r_fighter: 2., // fitness benefit of fighter phenotype
        r_e: 1., // fitness benefit of quality
        r_baseline: 5., // baseline fitness
        u_q: vec![0.;N], //(n) survivals from maintenance
        a_q: vec![0.;N], //(n) arrivals from dispersal
        mu_e: 0.1, // Rate of death given lack of energy reserves
        mu_base: 0.1, // Baseline rate of death
        r_q: vec![0.;N], //(n) arrivals from local births
        R: vec![vec![0.;Kappa];N_states], //(s,i,e) fecundity of individual for each state
        rhon: vec![vec![0.;N];N_states], //(idx,n) prob. distribution over states for individuals in patches of density qi
        rhotilde: vec![0.;N_states], //(idx) state distribution of newborns
        L_g_n: vec![vec![0.;N];G], // L(g|n) likelihood of food mass g given patch density n
        pipig: vec![vec![vec![0.;G];I];I], //Trans. prob. og pi prime given pi and g [piprime][pi][g]
        x: vec![Kappa/2;N_states], // (idx) gives resident strategy -> threshold value above which input energy goes to quality
        dev_m: vec![vec![vec![0.;Kappa];N_states];N_states], // P(idx prime | idx, kappa) developmental transitions
        M: vec![vec![vec![0.;Kappa];N_states];N_states], // P(idx prime | idx, kappa) transitions from full growth cycle
        c_max: 0., // maximum maintenance investment
        lambda: 0.1, // Dispersal rate
        rhohat: vec![vec![0.;N_states];N], // state distribution after development fatalities, before normalisation
        gammah: 0.05, // parameter determining trans. prob high to low density patch, due to habitat degradation and death of individuals
        gammal: 0.0, // parameter determining trans. prob low to high density patch, due to narrowing patch habitat quality
        tol: 0.001, // tolerance threshold for difference in demographic state, below which projections end
        p_stoch: vec![vec![0.;N];N], // transition probabilities of patches given demo stoch and perturbations
        pno: vec![vec![0.;N];N], // (n_prime,n) latent transition prob. of patches between cycles
        pdet: vec![vec![0.;N];N], // (n_prime,n) latent transition prob. of patches between cycles given deterministic processes
        x_prime: vec![Kappa-1;N_states], // Best response strategy
        ptilde: vec![vec![0.;N];N], // (n_prime,n) Evolved prior for transition prob. of patches at demographic stability
        V: vec![vec![0.;N_states];2], // (n, state) Fitness
        tolv: 0.1, // tolerance threshold in the difference in v for finding the best response
        pipitime: vec![vec![0.;I];I], // (piprime,i) prob of belief transition given time
        p_g_i: vec![vec![0.;I];G], // (g,i) the probability of food mass g given prior i
        pert_survival: vec![0.;I], // (i) probability of survival given prior belief i
        pceeg: vec![vec![vec![0.;G];E];E], // probability of transitioning to eprime given e and g
        pcsurvival: vec![0.;N_states], // probability of survival given reserves state and state dependent strategy
    };

    // find ess strategy and demographic stability
    env0.run(&policypath);
    let mut r = RSession::new()?;
    println!("Running R script to plot ESS strategy...");
    r.exec(&format!("path <- '{}'", projectpath))?;
    r.exec("source('src/sim_plot.r')")?;
    


    // life history projection with natural patch dynamics
    println!("Running life history projection with natural patch dynamics...");
    let pnatural = env0.pno.clone();
    env0.lifehistory(&LHpath1, pnatural);
    
    // plot
    r.exec(&format!("path <- '{}'", path2))?;
    r.exec("source('src/LifeHistory.r')")?;



    // life history projection with lab patch dynamics (e.g. fixed patch sizes)
    println!("Running life history projection with lab patch dynamics...");
    let mut plab = vec![vec![0.;2];2];
    plab[1][1] = 1.;
    plab[0][0] = 1.;
    env0.lifehistory(&LHpath2, plab);
    
    // plot
    r.exec(&format!("path <- '{}'", path3))?;
    r.exec("source('src/LifeHistory.r')")?;

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