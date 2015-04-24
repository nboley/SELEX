// Argument parsing
use std::env;

use std::error::Error;

// File parsing
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;

// Random number generation
extern crate rand;
use rand::distributions::{IndependentSample, Range};

static R: f32 = 0.8314472;
static T: f32 = 300.0;
static NUM_SIMS_PER_RND: i32 = 100000;

struct Score {
    a: f32,
    c: f32,
    g: f32,
    t: f32
}

#[derive(Debug)]
enum Base{ A, C, G, T, N }

#[derive(Debug)]
struct SelexRead {
    seq: Vec<Base>,
    round: i32,
    cnt: i32,
}

/*
fn update_weights(rd: &SelexRead, 
                  base_scores: &Vec<Score>,
                  unbnd_conc: f32) -> Vec<f32> {
    let bs_iter = rd.seq.windows(base_scores.len());
    let mut weights = Vec::with_capacity(bs_iter.len());

    let mut total_occupancy = 0.0;

    for substr in bs_iter {
        let occupancy = calc_occupancy(substr, base_scores, unbnd_conc);
        total_occupancy += occupancy;
        weights.push( occupancy );
    }
    
    for weight in weights.iter_mut() {
        *weight /= total_occupancy;
    }
    
    weights
}

#[derive(Debug)]
struct SelexReadAndWeights {
    rd: SelexRead,
    weights: Vec<f32>,
}
*/

fn load_sequences_in_round(fname: &str, round: i32) -> Vec<SelexRead> {
    let file = match File::open(&fname) {
        // The `desc` field of `IoError` is a string that describes the error
        Err(why) => panic!(
            "couldn't open {}: {}", fname, Error::description(&why) ),
        Ok(file) => file,
    };

    let mut seqs: Vec<SelexRead> = Vec::new();
    
    for line in BufReader::new(&file).lines() {
        let res = line.unwrap();
        let mut line_iter = res.split("\t");

        let mut seq = Vec::new();
        for char in line_iter.next().unwrap().chars() {
            let base = match char {
                'A' => Base::A,
                'C' => Base::C,
                'G' => Base::G,
                'T' => Base::T,
                _ => Base::N,
            };
            seq.push(base); 
        }

        let cnt: i32  = line_iter.next().unwrap_or("1").parse().unwrap_or(1);
        let rd = SelexRead{seq: seq, round: round, cnt: cnt};
        seqs.push(rd);
    };

    seqs
}


fn calc_gibbs_free_energy(seq: &[Base], base_scores: &Vec<Score>) -> f32 {
    assert!(seq.len() == base_scores.len());
    let mut score: f32 = 0.0;
    for pos in 0..base_scores.len() {
        score += match seq[pos] {
            Base::A => base_scores[pos].a,
            Base::C => base_scores[pos].c,
            Base::G => base_scores[pos].g,
            Base::T => base_scores[pos].t,
            Base::N => unreachable!(),
        };
    }    
    score 
}

fn calc_occupancy(seq: &[Base], 
                  base_scores: &Vec<Score>, 
                  unbnd_conc: f32) -> f32 { 
    let gfe = calc_gibbs_free_energy(seq, base_scores);
    let numerator = unbnd_conc*(-gfe/(R*T)).exp();
    numerator/(1.0+numerator)
}


fn calc_max_occupancy(seq: &[Base], 
                      base_scores: &Vec<Score>,
                      unbnd_conc: f32) -> f32 {
    let bs_iter = seq.windows(base_scores.len());
    let mut max_occupancy = 0.0;
    
    for substr in bs_iter {
        let occupancy = calc_occupancy(substr, base_scores, unbnd_conc);
        if occupancy > max_occupancy {
            max_occupancy = occupancy;
        };
    }
    
    max_occupancy
}


fn simulate_read(len: usize) -> Vec<Base> {
    let mut rng = rand::thread_rng();
    let range_iter = Range::new(0, 4);
    let mut seq = Vec::with_capacity(len);
    for _ in 0..len {
        let base = match range_iter.ind_sample(&mut rng) {
            0 => Base::A,
            1 => Base::C,
            2 => Base::G,
            3 => Base::T,
            _ => unreachable!()
        };
        seq.push(base);
    }
    seq
    //println!("{:?}", seq);
}

fn load_all_reads() -> Vec<SelexRead> {
    let mut observed_reads: Vec<SelexRead> = Vec::new();
        
    for fname in env::args().skip(1) {
        let round: i32 = 
            fname.split("_").last().unwrap(
                ).split(".").next().unwrap(
                ).parse().unwrap();
        
        for rd in load_sequences_in_round(&fname, round) {
            observed_reads.push( rd );
        }

        print!("Loaded {} (Round {})\n", fname, round);
    };
    
    observed_reads
}

/*
fn load_uniform_score_matrix(motif_len: i32) -> Vec<Score> {
    let mut score_matrix = Vec::with_capacity(motif_len as usize);
    for _ in 0..motif_len {
        score_matrix.push(Score{a: -1.0, c: -1.0, g: -1.0, t: -1.0})
    }
    score_matrix
}
*/

fn load_initial_score_matrix() -> Vec<Score> {
    let motif_len = 6;
    let mut score_matrix = Vec::with_capacity(motif_len as usize);
    score_matrix.push(Score{a: -1.0, c: -1.0, g: -1.0, t: -10.0});
    score_matrix.push(Score{a: -10.0, c: -1.0, g: -1.0, t: -1.0});
    score_matrix.push(Score{a: -10.0, c: -1.0, g: -1.0, t: -1.0});
    score_matrix.push(Score{a: -1.0, c: -1.0, g: -1.0, t: -1.00});
    score_matrix.push(Score{a: -1.0, c: -5.0, g: -1.0, t: -1.0});
    score_matrix.push(Score{a: -1.0, c: -5.0, g: -1.0, t: -1.0});

    score_matrix
}

fn estimate_normalization_constants(
    obs_rds: &Vec<SelexRead>, 
    score_mat: &Vec<Score>, 
    unbnd_concs: &Vec<f32>
        ) -> f32 {
    // Find the number of reads in each round
    let mut rnd_cnts = vec![0; unbnd_concs.len()];
    for rd in obs_rds.iter() {
        rnd_cnts[rd.round as usize] += 1;
    };
    
    // Initialize a vector to hold the normalization factors
    let mut norm_constants = vec![0.0; unbnd_concs.len()];

    // Loop through the observed round 1 reads to make sure that we get the seq 
    // length distribution correct
    let mut num_sim_reads = 0;
    while num_sim_reads < NUM_SIMS_PER_RND {
        for rd in obs_rds.iter() {
            num_sim_reads += 1;
            
            let sim_seq = simulate_read(rd.seq.len());
            let mut curr_occupancy = 1.0;
            for (rnd, unbnd_conc) in unbnd_concs.iter().enumerate() {
                curr_occupancy *= calc_max_occupancy(
                    &(sim_seq[..]), &score_mat, *unbnd_conc);
                norm_constants[rnd] += curr_occupancy;
            };
            if num_sim_reads >= NUM_SIMS_PER_RND { break; }
        };
    };
    
    let mut norm_constant = 0.0;
    for (i, constant) in norm_constants.iter().enumerate() {
        norm_constant += constant*(rnd_cnts[i] as f32)/(NUM_SIMS_PER_RND as f32 );
    };
    
    norm_constant
}

fn calc_lhd(
    obs_rds: &Vec<SelexRead>, 
    score_mat: &Vec<Score>, 
    unbnd_concs: &Vec<f32>
        ) -> f32
{
    let mut lhd_num = 0.0; 
    for rd in obs_rds.iter() {
        let unbnd_conc = unbnd_concs[rd.round as usize];
        let log_occ: f32 = calc_max_occupancy(
            &(rd.seq[..]), &score_mat, unbnd_conc).log(2.0);
        lhd_num += (rd.round as f32)*log_occ;
    }

    let norm_constant = estimate_normalization_constants(
        &obs_rds, &score_mat, &unbnd_concs);

    lhd_num - norm_constant
}

/*
TODO:
0) get better dg mnatrix
1) implement simulator
2) use reverse complement

*/
 
fn main() {
    let score_matrix = load_initial_score_matrix();
        
    let observed_reads = load_all_reads();

    let unbnd_concentrations = vec![1.0; env::args().len()];

    for _ in 1..10 {
        let lhd = calc_lhd(
            &observed_reads, &score_matrix, &unbnd_concentrations);
        print!("Lhd: {}\n", lhd);
    }
}
