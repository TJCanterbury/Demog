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

fn init_pop(n: u32, agent: Agent) -> Vec<Agent> {
    // returns vector with new agents
    let mut pop = Vec::new();
    for _i in 0..n {
        pop.push(agent.clone());
    }
    return pop;
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
    P_stay: f64,
    P_change: f64,
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
    s * NPI * NM + pi * NM + m
}

impl Environment { 
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

    fn interpolate_pi(&self, mut prime:f64) -> (usize,usize,f64,f64) {
        prime = prime.min(1.0).max(0.0);
        let d1 = ((NPI as f64) * prime).floor();
        let d2 = d1 + 1.0;
        let p2 = ((NPI as f64) * prime) - d1;
        let p1 = 1.0 - p2;

        return (d1 as usize,d2 as usize,p1,p2)
    }

    fn update_r(&mut self){
        for i in 0..self.I {
            self.r_qi[i] = 0.0;

            for s in 0..self.S{
                for pi in 0..self.PI{
                    for m in 0.. self.M{
                        self.r_qi[i] += self.rhoi[s][pi][m][i]*self.R[s][m][i]
                    }
                }
            }
        }
    }

    fn newborns(&mut self){
        self.rhotilde.fill(0.0);
        
        let mut approx = 0.;

        for i in 0..self.I/2{
            approx += self.hi[i]; // Equation 3
        }

        let (d1, d2, p1, p2) = self.interpolate_pi(approx);
        
        //Equations 4 & 5
        self.rhotilde[idx(0,d1,0)] = p1;
        self.rhotilde[idx(0,d2,0)] = p2;
    }

    fn pop_growth(&mut self){

        let mut rhoiprime: Vec<Vec<f64>> = vec![vec![0.0;self.I]; NM * NPI * NS];
         // Equation 8
        let dispersing_pool: f64 = (0..self.I)
                .map(|j| self.hi[j]*self.q(j)*self.r_qi[j])
                .sum();

        for i in 0..self.I {
            // equation 7
            self.a_qi[i] = (1. - self.mu)*self.q(i)*self.r_qi[i] + self.mu*self.hi[i]*dispersing_pool;
             // Equation 9
            let qiprime: f64 = self.a_qi[i] + self.q(i);

             // Equations 10-13
            let (d1, d2, p1, p2) = self.interpolate_i(qiprime, 0.0, self.q(self.I));
            self.p_births[i].fill(0.0); 
            self.p_births[i][d1] += p1;
            self.p_births[i][d2] += p2;
            
            // Individual state distribution update
            for s in 0..self.S {
                for pi in 0..self.PI {
                    for m in 0..self.M {
                        let pi_prime = pi * self.P_stay + (1. - pi) * self.P_change;
                        let (dpi1, dpi2, ppi1, ppi2) = self.interpolate_pi(pi_prime);

                         //Equations 14 and 15
                        rhoiprime[idx(s,dpi1,m)][d1] += ppi1 * p1 * (((i as f64)/(d1 as f64)) * self.rhoi[idx(s,pi,m)][i] + (((d1 as f64)-(i as f64))/(d1 as f64)) * self.rhotilde[idx(s,pi,m)]);
                        rhoiprime[idx(s,dpi1,m)][d2] += ppi1 * p2 * (((i as f64)/(d2 as f64)) * self.rhoi[idx(s,pi,m)][i] + (((d2 as f64)-(i as f64))/(d2 as f64)) * self.rhotilde[idx(s,pi,m)]);
                        rhoiprime[idx(s,dpi2,m)][d1] += ppi2 * p1 * (((i as f64)/(d1 as f64)) * self.rhoi[idx(s,pi,m)][i] + (((d1 as f64)-(i as f64))/(d1 as f64)) * self.rhotilde[idx(s,pi,m)]);
                        rhoiprime[idx(s,dpi2,m)][d2] += ppi2 * p2 * (((i as f64)/(d2 as f64)) * self.rhoi[idx(s,pi,m)][i] + (((d2 as f64)-(i as f64))/(d2 as f64)) * self.rhotilde[idx(s,pi,m)]);
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

    fn init_obs_matrix(&mut self){
        let mut piprime;
        for pi in 0..NPI{
            for g in 0..self.G{
                piprime = (pi as f64 * self.p_g_i[g][pi])/();
                self.obs_matrix[piprime1][pi][g] += p1;
                self.obs_matrix[piprime2][pi][g] += p2;
            }
        }
    }

    fn update_P_staychange(&mut self){
        //
    }


}


// main Functions
fn main() -> std::io::Result<()>  {
// Comand-line arguments
    let args: Vec<String> = env::args().collect();
    let project_id = &args[1];
    let path = format!("./Results/{}/", project_id);
    // Construct the full path
    let path_construct = Path::new(&path);
    // Ensure parent directory exists
    let _ = fs::create_dir_all(path_construct); // Create directory path if it doesn't exist
    let _ = write_csv_header(&path);

/////////////////////////////////////////// Initialise parameters \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    let env0 = Environment {
        pop: init_pop(size, agent0),
        habitat: 0,
        singles: (0..size as usize).collect(),
        dead: Vec::new(),
        mean_fitness: 0.,
        d: 1.0, // winter death rate
        mu: 0.01,
        stability: 0.,
        prop_obs: 0.,
        climate_match:0.,
        obs_c:0.25,
        stoch:0.5,
        theta_low:0.25,
        theta_high:0.75,
        sigma_mass:1.0,
        cli_sigma:2.,
        days:20.,
        turns:T,
        p:0.8,
        mut_size:0.2,
        div_rate:0.,
        mean_mass:0.,
        cue_error:0.2
    };

// start r session
     let mut r = RSession::new()?;
    r.exec("source('src/plots.r')")?;
    let r_mutex = Mutex::new(r);

// climate stochasticity: 
    let path = format!("./Results/{}/stoch/", project_id);
    let path_construct = Path::new(&path);
    // Ensure parent directory exists
    let _ = fs::create_dir_all(path_construct); // Create directory path if it doesn't exist
    let _ = write_csv_header(&path);
    (0..iterations).into_par_iter().for_each(|g|  {
        // Initialise stochastic variables
        let mut rng = rand::thread_rng();
        let habitat;
        if rng.gen::<f64>() < 0.5 {
            habitat = 1;
        } else {
            habitat = 0;
        }
        let mut env = env0.clone();
        let mut agent = agent0.clone();
        agent.mutate(1.0, 1.0); // randomize resident loci
        env.pop = init_pop(1000, agent);   
        env.habitat = habitat;
        let x = rng.gen_range(0.0..1.0); // uniform sample from parameter space
        env.stoch = x;
        
        println!("Simulation started: stoch: {}, trial: {}", x, g);
        run(
            generations, 
            &path, 
            Some(&(x.to_string()+"_stoch_")), 
            Some(&g),
            env
        );
        println!("Simulation done: stoch: {}, trial: {}", x, g);
        if g % 5 == 0 {
            let mut r_guard = r_mutex.lock().unwrap();
            r_guard.exec(&format!("run_stoch_plot('{}')", path)).unwrap();
        }
    });
    let mut r = RSession::new()?;
    r.exec(&format!("run_stoch_plot('{}')", path)).unwrap();

// // Harshness:
//     let path = format!("./Results/{}/theta/", project_id);
//     let path_construct = Path::new(&path);
//     // Ensure parent directory exists
//     let _ = fs::create_dir_all(path_construct); // Create directory path if it doesn't exist
//     let _ = write_csv_header(&path);
//     (0..iterations).into_par_iter().for_each(|g|  {
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
//         let x = rng.gen_range(0.0..0.5);
//         env.theta_low = 0.5-x;
//         env.theta_high = 0.5+x;
//         println!("Simulation started: theta: {}, trial: {}", x, g);
//         run(
//             generations, 
//             &path, 
//             Some(&(x.to_string()+"_theta_")), 
//             Some(&g),
//             env
//         );
//         println!("Simulation done: theta: {}, trial: {}", x, g);
//         if g % 5 == 0 {
//             let mut r_guard = r_mutex.lock().unwrap();
//             r_guard.exec(&format!("run_theta_plot('{}')", path)).unwrap();
//         }
//     });
//     let mut r = RSession::new()?;
//     r.exec(&format!("run_theta_plot('{}')", path)).unwrap();

// // p:
//     let path = format!("./Results/{}/p/", project_id);
//     let path_construct = Path::new(&path);
//     // Ensure parent directory exists
//     let _ = fs::create_dir_all(path_construct); // Create directory path if it doesn't exist
//     let _ = write_csv_header(&path);
//     (0..iterations).into_par_iter().for_each(|g|  {
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
//         let x = rng.gen_range(0.5..1.0);
//         env.p = x;
//         println!("Simulation started: p: {}, trial: {}", x, g);
//         run(
//             generations, 
//             &path, 
//             Some(&(x.to_string()+"_p_")), 
//             Some(&g),
//             env
//         );
//         println!("Simulation done: p: {}, trial: {}", x, g);
//         if g % 5 == 0 {
//             let mut r_guard = r_mutex.lock().unwrap();
//             r_guard.exec(&format!("run_p_plot('{}')", path)).unwrap();
//         }
//     });
//     let mut r = RSession::new()?;
//     r.exec(&format!("run_p_plot('{}')", path)).unwrap();

// // Social cue noise 
//     let path = format!("./Results/{}/cue_noise/", project_id);
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
//         let x = rng.gen_range(0.0..0.5); // uniform sample from parameter space
//         env.cue_error = x;
        
//         println!("Simulation started: cue_noise: {}, trial: {}", x, g);
//         run(
//             generations, 
//             &path, 
//             Some(&(x.to_string()+"_cue_noise_")), 
//             Some(&g),
//             env
//         );
//         println!("Simulation done: cue_noise: {}, trial: {}", x, g);
//         if g % 5 == 0 {
//             let mut r_guard = r_mutex.lock().unwrap();
//             r_guard.exec(&format!("run_cue_noise_plot('{}')", path)).unwrap();
//         }
//     });
//     let mut r = RSession::new()?;
//     r.exec(&format!("run_cue_noise_plot('{}')", path)).unwrap();


// // Cost:
//     let path = format!("./Results/{}/cost/", project_id);
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
//         env.obs_c = x;
//         println!("Simulation started: cost: {}, trial: {}", x, g);
//         run(
//             generations, 
//             &path, 
//             Some(&(x.to_string()+"_cost_")), 
//             Some(&g),
//             env
//         );
//         println!("Simulation done: cost: {}, trial: {}", x, g);
//         if g % 5 == 0 {
//             let mut r_guard = r_mutex.lock().unwrap();
//             r_guard.exec(&format!("run_cost_plot('{}')", path)).unwrap();
//         }
//     });
//     let mut r = RSession::new()?;
//     r.exec(&format!("run_cost_plot('{}')", path)).unwrap();

    
// // climate noise: 
//     let path = format!("./Results/{}/cli/", project_id);
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
//         let x = rng.gen_range(0.05..5.0); // uniform sample from parameter space
//         env.cli_sigma = x;
        
//         println!("Simulation started: cli: {}, trial: {}", x, g);
//         run(
//             generations, 
//             &path, 
//             Some(&(x.to_string()+"_cli_")), 
//             Some(&g),
//             env
//         );
//         println!("Simulation done: cli: {}, trial: {}", x, g);
//         if g % 5 == 0 {
//             let mut r_guard = r_mutex.lock().unwrap();
//             r_guard.exec(&format!("run_cli_plot('{}')", path)).unwrap();
//         }
//     });
//     let mut r = RSession::new()?;
//     r.exec(&format!("run_cli_plot('{}')", path)).unwrap();

    
// G: 
    let path = format!("./Results/{}/G/", project_id);
    let path_construct = Path::new(&path);
    // Ensure parent directory exists
    let _ = fs::create_dir_all(path_construct); // Create directory path if it doesn't exist
    let _ = write_csv_header(&path);
    (0..iterations).into_par_iter().for_each(|g|  {
        // Initialise stochastic variables
        let mut rng = rand::thread_rng();
        let habitat;
        if rng.gen::<f64>() < 0.5 {
            habitat = 1;
        } else {
            habitat = 0;
        }
        let mut env = env0.clone();
        let mut agent = agent0.clone();
        agent.mutate(1.0, 1.0); // randomize resident loci
        env.pop = init_pop(1000, agent);   
        env.habitat = habitat;
        let x = rng.gen_range(1.0..100.0); // uniform sample from parameter space
        env.days = x;
        
        println!("Simulation started: G: {}, trial: {}", x, g);
        run(
            generations, 
            &path, 
            Some(&(x.to_string()+"_cli_")), 
            Some(&g),
            env
        );
        println!("Simulation done: G: {}, trial: {}", x, g);
        if g % 5 == 0 {
            let mut r_guard = r_mutex.lock().unwrap();
            r_guard.exec(&format!("run_G_plot('{}')", path)).unwrap();
        }
    });
    let mut r = RSession::new()?;
    r.exec(&format!("run_G_plot('{}')", path)).unwrap();

    Ok(())
}