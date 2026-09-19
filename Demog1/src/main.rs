use libm::log2;
use rand::Rng;
use rand::thread_rng;
use rand::seq::SliceRandom;
use rand_distr::{Normal, Distribution};
use statrs::distribution::{Normal as nm, Continuous as cn};
use weighted_rand::builder::*;
extern crate csv;
use std::error::Error;
use std::fs::OpenOptions;
use std::intrinsics::floorf64;
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
const NS: usize = 10;
const NPI: usize = 10;
const NM: usize = 10;
const LENPI: usize = (NPI-1)*(NPI-2)/2;

#[derive(Clone, Debug)]
struct Environment {
    hi: Vec<f64>, //(i) pdf over patch population densities
    n_max: f64,
    I: usize,
    G: usize,
    S: usize,
    PI: usize,
    M: usize,
    g_mean: f64,
    g_sd: f64
    r_fighter: f64,
    r_baseline: f64,
    a_qi: Vec<f64>, //(i) arrivals from dispersal
    mu: f64, // Dispersal rate
    r_qi: Vec<f64>, //(i) arrivals from local births
    R: Vec<Vec<f64>>, //(s,m) fecundity of individual of size s and maturity m
    rhoi: Vec<Vec<f64>>, //(idx,i) prob. distribution over states for individuals in patches of density qi
    p_births: Vec<Vec<f64>>, //(i,j)
    rhotilde: Vec<f64>, //(idx)
    p_g_i: Vec<Vec<f64>>,
    pipig: Vec<Vec<Vec<f64>>, //Trans. prob. og pi prime given pi and g [piprime][pi][g]
    Beliefs: Vec<BeliefState>,
}

#[derive(Clone, Debug)]
struct BeliefState {
    pi0: usize,
    pi1: usize,
    pi2: usize,
}

// Implementations
fn bernouli(prob:f64) -> f64 {
    let dice = thread_rng().gen::<f64>();
    if dice < prob {
        return 1.;
    } else {
        return 0.;
    }
}

fn idx(s:usize, pi:usize, m:usize) -> usize {
    s * LENPI * NM + pi * NM + m
}

fn vectorise_pi(n0:usize,n1:usize)->usize{
    // first term tells you which row (n0) if it were a 2x2 grid, 
    // second term accounts for missing lower triangle given simplex
    // third term gives the column (n1)
    return n0*(NPI+1) - (n0*(n0-1))/2 + n1
}

impl Environment { 
    fn init_beliefs(&mut self){
        // Run at the beginning to initialise the vector of belief states
        for n0 in 0..NPI{
            for n1 in 0..NPI{
                if NPI >= n0+n1{
                    let j = vectorise_pi(n0, n1);
                    self.Beliefs[j].pi0=n0;
                    self.Beliefs[j].pi1=n1;
                    self.Beliefs[j].pi2=NPI-n0-n1;
                }
            }
        }
    }

    fn q(&self, i: usize)->f64{
        return self.n_max*(i as f64)/(self.I as f64)
    }

    fn interpolate_i(&self, mut prime:f64, min_val: f64, max_val: f64) -> (usize,usize,f64,f64) {
        prime = prime.min(max_val).max(min_val);
        let d1 = (self.I as f64 * prime/self.n_max).floor();
        let d2 = d1 + 1.0;
        let p2 = (self.I as f64 * prime/self.n_max) - d1;
        let p1 = 1.0 - p2;

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

    fn update_r(&mut self){
        for i in 0..self.I {
            self.r_qi[i] = 0.0;

            for s in 0..self.S{
                for pi in 0..self.PI{
                    for m in 0..self.M{
                        self.r_qi[i] += self.rhoi[s][pi][m][i]*self.R[s][m][i]
                    }
                }
            }
        }
    }

    fn newborns(&mut self){
        self.rhotilde.fill(0.0);

        // interpolating priors onto the grid
        let (d1, d2, d3, f1, f2, f3) = self.decompose_pi(self.hi[0],self.hi[1],self.hi[2]);
        let mut pi;
        if (f1+f2+f3) as usize == 0{

            pi = vectorise_pi(d1, d2);
            self.rhotilde[idx(0,pi,0)] = 1.;

        } else if (f1+f2+f3) as usize == 1{

            pi = vectorise_pi(d1+1, d2);
            self.rhotilde[idx(0,pi,0)] += f1;

            pi = vectorise_pi(d1, d2+1);
            self.rhotilde[idx(0,pi,0)] += f2;

            pi = vectorise_pi(d1, d2);
            self.rhotilde[idx(0,pi,0)] += f3;
            
        } else if (f1+f2+f3) as usize == 2{

            pi = vectorise_pi(d1, d2+1);
            self.rhotilde[idx(0,pi,0)] += 1.-f1;

            pi = vectorise_pi(d1+1, d2);
            self.rhotilde[idx(0,pi,0)] += 1.-f2;

            pi = vectorise_pi(d1+1, d2+1);
            self.rhotilde[idx(0,pi,0)] += 1.-f3;
 
        }
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
        let mut rhoiprime: Vec<Vec<f64>> = vec![vec![0.0;self.I]; NM * NPI * NS];

        // 3.2.2 Patch projection
        // Second expression of equation 4 defining alpha, representing the number of migrants
        let dispersing_pool: f64 = (0..self.I)
                .map(|j| self.hi[j]*self.q(j)*self.r_qi[j])
                .sum();

        for i in 0..self.I {
            // calculating alpha, equation 4
            self.a_qi[i] = (1. - self.mu)*self.q(i)*self.r_qi[i] + self.mu*dispersing_pool;

            // calculating the new patch density size after dispersal, equation 6
            let qiprime: f64 = self.a_qi[i] + self.q(i);

            // Interpolating iprime to find transition probabilities on 1-simplex qi grid, equations 7-10
            let (d1, d2, p1, p2) = self.interpolate_i(qiprime, 0.0, self.q(self.I));
            self.p_births[i].fill(0.0); 
            self.p_births[i][d1] += p1;
            self.p_births[i][d2] += p2;
        }
            
        // Now we move on to projection of the patch density distribution 
        // and of physiological states:
        for i in 0..self.I {
            // Physiological state distribution update
            for s in 0..self.S {
                for pi in 0..self.PI {
                    for m in 0..self.M {
                        for j in 0..self.I{
                            rhoiprime[idx(s,pi,m)][i] += self.hi[j]*self.p_births[j][i]*((j as f64/i as f64)*self.rhoi[idx(s,pi,m)][j] + ((i as f64-j as f64)/i as f64)*self.rhotilde[idx(s,pi,m)]);
                        }
                    }
                }
            }
        }
        for i in 0..self.I {
            // make rhoi rhoiprime
            for s in 0..self.S {
                for pi in 0..self.PI {
                    for m in 0..self.M {
                        self.rhoi[idx(s,pi,m)][i] = rhoiprime[idx(s,pi,m)][i];
                    }
                }
            }
        }

        // Patch state distribution update
        let mut h_prime: Vec<f64> = vec![0.0;self.I];
        for j in 0..self.I {
            for i in 0..self.I {
                h_prime[i] += self.hi[j] * self.p_births[j][i]; // Equation 13
            }
        }
        for i in 0..self.I {
            self.hi[i] = h_prime[i];
        }

    }

    fn init_p_g_i(&mut self){
        // This function updates the probability of a food mass of g in a patch of density i
        // as per the first equaitions of section 3.3 defining the pdf over food masses.
        let mut q_i:f64;
        for i in 0..self.I{
            q_i = self.q(i);
            // Equation 18
            let K: f64 = (0..self.G)
                .map(|g| (-((g as f64-self.g_mean/q_i).powf(2.))/(2.*self.g_sd.powf(2.))).exp())
                .sum();

            // Equation 17
            for g in 0..self.G {
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
        for vec in self.pipig {
            for vec2 in vec {
                vec2.fill(0.);
            }
        }
        for pi in 0..self.PI{
            for g in 0..self.G{
                // extract belief distribtion from flat pi vertex index
                pi1 = self.Beliefs[pi].pi0 as f64;
                pi2 = self.Beliefs[pi].pi1 as f64;
                pi3 = self.Beliefs[pi].pi2 as f64;

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

    fn Foraging(&mut self){
        // This function will do all the updates associates with foraging and development
        // as per section 3.3 in finding demographic stability.

        // first update food mass pdf:
        self.init_p_g_i();

        // Initialise the transition probabilities from pi to pi^prime given a food mass of g
        self.init_pipig();

        // Development
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

// /////////////////////////////////////////// Initialise parameters \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
//     let env0 = Environment {
//         pop: init_pop(size, agent0),
//         habitat: 0,
//         singles: (0..size as usize).collect(),
//         dead: Vec::new(),
//         mean_fitness: 0.,
//         d: 1.0, // winter death rate
//         mu: 0.01,
//         stability: 0.,
//         prop_obs: 0.,
//         climate_match:0.,
//         obs_c:0.25,
//         stoch:0.5,
//         theta_low:0.25,
//         theta_high:0.75,
//         sigma_mass:1.0,
//         cli_sigma:2.,
//         days:20.,
//         turns:T,
//         p:0.8,
//         mut_size:0.2,
//         div_rate:0.,
//         mean_mass:0.,
//         cue_error:0.2
//     };

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