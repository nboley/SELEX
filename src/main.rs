#![allow(dead_code)]

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

// kJ/mol*K
static R: f32 = 8.314472e-3;
// Kelvin
static T: f32 = 300.0;
static NUM_SIMS_PER_RND: i32 = 100000;

#[derive(Debug)]
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

#[derive(Debug)]
struct ObservedReads {
    rds: Vec<SelexRead>,
    max_rnd: usize
}

#[derive(Debug)]
struct BindingModel {
    factor: String,
    consensus_gfe: f32,
    pos_gfes: Vec<Score>
}


/*
fn load_uniform_score_matrix(motif_len: i32) -> Vec<Score> {
    let mut score_matrix = Vec::with_capacity(motif_len as usize);
    for _ in 0..motif_len {
        score_matrix.push(Score{a: -1.0, c: -1.0, g: -1.0, t: -1.0})
    }
    score_matrix
}

fn load_initial_binding_model() -> BindingModel {
    let mut bm = BindingModel{ 
        consensus_gfe: 40.0, pos_gfes: Vec::new() };
    bm.pos_gfes.push(Score{a: -10.0, c: -10.0, g: -10.0, t: 0.0});
    bm.pos_gfes.push(Score{a: 0.0, c: -10.0, g: -10.0, t: -10.0});
    bm.pos_gfes.push(Score{a: 0.0, c: -10.0, g: -10.0, t: -10.0});
    bm.pos_gfes.push(Score{a: -10.0, c: -10.0, g: -10.0, t: 0.00});
    bm.pos_gfes.push(Score{a: -2.0, c: 0.0, g: -2.0, t: -2.0});
    bm.pos_gfes.push(Score{a: -2.0, c: 0.0, g: -2.0, t: -2.0});
    bm.pos_gfes.push(Score{a: -2.0, c: 0.0, g: -2.0, t: -2.0});
    bm.pos_gfes.push(Score{a: -2.0, c: 0.0, g: -2.0, t: -2.0});

    bm
}


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

fn calc_unbnd_conc_for_specified_max_occupancy(
    bm: &BindingModel, consensus_occ: f32) -> f32 {
// Given a binding model, find what the concensus energy and 
// unbound concentration would need to be to produce the 
// specified occupancies
    let concensus_seq = calc_consensus_seq(bm);
    let (fwd_gfe, rc_gfe) = calc_fwd_and_revcmp_gibbs_free_energy(
        &concensus_seq, bm );
    let mut gfe = fwd_gfe;
    if rc_gfe > gfe { gfe = rc_gfe };

    let norm_gfe = gfe/(R*T);
    let unbnd_conc = (consensus_occ/(1.0-consensus_occ))*norm_gfe.exp();

    let occupancy = calc_max_occupancy(&concensus_seq, bm, unbnd_conc);

    print!("GFE: {}\tUnbnd Conc: {}\tCalc Occ:{}\n", 
           gfe, unbnd_conc, occupancy);
    unbnd_conc
}
*/

fn logit(x: f32) -> f32 {
    x.ln() - (1.0-x).ln()
}

fn biased_logit(x: f32, bias: f32) -> f32 {
    logit(bias/2.0 + (1.0-bias)*x)
}

fn logistic(x: f32) -> f32 {
    x.exp()/(1.0 + x.exp())
}

fn load_motif(fname: &str) -> BindingModel {
    let file = match File::open(&fname) {
        Err(why) => panic!(
            "couldn't open {}: {}", fname, Error::description(&why) ),
        Ok(file) => file,
    };

    let mut factor = String::new();
    let mut consensus_gfe = 0.0;
    let mut pos_gfes = Vec::new();
    for line in BufReader::new(&file).lines() {
        // unwrap the line
        let line = line.unwrap();
        // if it starts with a > this is a factor, so save the factor name
        if line.starts_with(">") {
            factor = line;
        } 
        // otherwise this should be an enerby list, so parse and save these
        else {
            let energies: Vec<f32> = line.split(" ").skip(1).map(
                |x| biased_logit(x.parse().unwrap(), 1e-6)).collect();
            let mut max_energy = energies[0];
            for energy in energies.iter().skip(1) {
                if *energy > max_energy { 
                    max_energy = *energy; 
                }
            }
            consensus_gfe += max_energy;
            let pos_scores = Score{ a: energies[0]-max_energy, 
                                    c: energies[1]-max_energy, 
                                    g: energies[2]-max_energy, 
                                    t: energies[3]-max_energy};
            pos_gfes.push( pos_scores );
        }
    }
    BindingModel{factor: factor, 
                 consensus_gfe: consensus_gfe, 
                 pos_gfes: pos_gfes}
}

fn load_sequences_in_round(fname: &str, round: i32) -> Vec<SelexRead> {
    let file = match File::open(&fname) {
        Err(why) => panic!(
            "couldn't open {}: {}", fname, Error::description(&why) ),
        Ok(file) => file,
    };

    let mut obs_rds = Vec::new();
    
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
        obs_rds.push(rd);
    };

    obs_rds
}

fn load_all_reads() -> ObservedReads {
    let mut observed_reads = ObservedReads{rds: Vec::new(), max_rnd: 0};
        
    for fname in env::args().skip(2) {
        let round: i32 = 
            fname.split("_").last().unwrap(
                ).split(".").next().unwrap(
                ).parse().unwrap();
        
        for rd in load_sequences_in_round(&fname, round) {
            observed_reads.rds.push( rd );
            if round as usize > observed_reads.max_rnd {
                observed_reads.max_rnd = round as usize;
            }
        }

        print!("Loaded {} (Round {})\n", fname, round);
    };
    
    observed_reads
}


fn calc_fwd_and_revcmp_gibbs_free_energy(
        seq: &[Base], bm: &BindingModel) -> (f32, f32) {
    assert!(seq.len() == bm.pos_gfes.len());
    let mut score: f32 = bm.consensus_gfe;
    let mut rc_score: f32 = bm.consensus_gfe;

    for pos in 0..bm.pos_gfes.len() {
        score += match seq[pos] {
            Base::A => bm.pos_gfes[pos].a,
            Base::C => bm.pos_gfes[pos].c,
            Base::G => bm.pos_gfes[pos].g,
            Base::T => bm.pos_gfes[pos].t,
            Base::N => (
                bm.pos_gfes[pos].a 
                + bm.pos_gfes[pos].c 
                + bm.pos_gfes[pos].g 
                + bm.pos_gfes[pos].t)/4.0,
        };

        rc_score += match seq[bm.pos_gfes.len() - pos - 1] {
            Base::A => bm.pos_gfes[pos].t,
            Base::C => bm.pos_gfes[pos].g,
            Base::G => bm.pos_gfes[pos].c,
            Base::T => bm.pos_gfes[pos].a,
            Base::N => (
                bm.pos_gfes[pos].a 
                    + bm.pos_gfes[pos].c 
                    + bm.pos_gfes[pos].g 
                    + bm.pos_gfes[pos].t)/4.0,
        };

    }    
    (score, rc_score)
}

fn calc_occupancy(gibbs_free_energy: f32, 
                  unbnd_conc: f32) -> f32 { 
    let numerator = unbnd_conc*(gibbs_free_energy/(R*T)).exp();
    numerator/(1.0+numerator)
}

fn calc_max_occupancy(seq: &[Base], 
                      bm: &BindingModel,
                      unbnd_conc: f32) -> f32 {
    let mut max_occupancy = 0.0;
    
    for sub_seq in seq.windows(bm.pos_gfes.len()) {
        let (gfe, rc_gfe) = calc_fwd_and_revcmp_gibbs_free_energy(
            sub_seq, bm);
        let fwd_occupancy = calc_occupancy(gfe, unbnd_conc);
        let rc_occupancy = calc_occupancy(rc_gfe, unbnd_conc);

        //print!("{:?}\t{}:{}\t{}:{}\n", sub_seq, gfe, fwd_occupancy, rc_gfe, rc_occupancy);

        if fwd_occupancy > max_occupancy {
            max_occupancy = fwd_occupancy;
        };

        if rc_occupancy > max_occupancy {
            max_occupancy = rc_occupancy;
        };
    }
    //print!("\n");
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
}


fn load_seq(seq_chars: &str) -> Vec<Base> {
    let mut seq = Vec::new();
    for char in seq_chars.chars() {
        let base = match char {
            'A' => Base::A,
            'C' => Base::C,
            'G' => Base::G,
            'T' => Base::T,
            _ => Base::N,
        };
        seq.push(base); 
    }
    seq
}

fn load_reads_in_fastq(fname: &str, round: i32) -> ObservedReads {
    let mut observed_reads = ObservedReads{ 
        rds: Vec::new(), max_rnd: round as usize };
    let fp = File::open(fname).unwrap();
    for (line_i, line) in BufReader::new(&fp).lines().enumerate() {
        if line_i%4 != 1 { continue; }
        let seq = load_seq(&(line.unwrap()));
        observed_reads.rds.push(SelexRead{seq: seq, round: round, cnt: 1});
    }
    observed_reads
}

fn estimate_average_occupancies(
    obs_rds: &Vec<SelexRead>, 
    bm: &BindingModel, 
    unbnd_concs: &Vec<f32>
        ) -> Vec<f32> {
    // Find the number of reads in each round
    let mut rnd_cnts = vec![0; unbnd_concs.len()];
    for rd in obs_rds.iter() {
        rnd_cnts[rd.round as usize] += 1;
    };
    
    // Initialize a vector to hold the normalization factors
    let mut total_occupancies = vec![0.0; unbnd_concs.len()];

    // Loop through the observed round 1 reads to make sure that we get the seq 
    // length distribution correct
    let mut num_sim_reads = 0;
    while num_sim_reads < NUM_SIMS_PER_RND {
        for rd in obs_rds.iter() {
            num_sim_reads += 1;
            
            let sim_seq = simulate_read(rd.seq.len());
            //print!("{:?}\t", sim_seq );

            let mut curr_occupancy = 1.0;
            for (rnd, unbnd_conc) in unbnd_concs.iter().enumerate() {
                let max_occ = calc_max_occupancy(
                    &(sim_seq[..]), &bm, *unbnd_conc);
                
                curr_occupancy *= max_occ;
                total_occupancies[rnd] += curr_occupancy;
                //print!("{}:{}\t", rnd, max_occ );
            };
            //print!("\n" );
            if num_sim_reads >= NUM_SIMS_PER_RND { break; }
        };
    };

    let mut mean_occupancies = Vec::new();
    
    for (i, val) in total_occupancies.iter().enumerate() {
        let prev_conc = if i == 0 {NUM_SIMS_PER_RND as f32} else { total_occupancies[i-1] };
        // print!("{}\t{}\t{}\t{}\n", i, prev_conc, val, val/prev_conc );
        //print!("Rnd: {}\tFrac Bnd: {}\n", i, val/prev_conc );
        mean_occupancies.push( val/prev_conc );
    }
    mean_occupancies
}

fn update_unbnd_concs(
    unbnd_concs: &mut Vec<f32>,
    obs_rds: &ObservedReads,
    bm: &BindingModel,     
    tot_DNA_conc: f32, 
    tot_prot_conc: f32 ) {

    let mean_occs = estimate_average_occupancies(
        &obs_rds.rds, bm, &unbnd_concs);
        
    for (rnd, mean_occ) in mean_occs.iter().enumerate() {
        let mut unbnd_conc = tot_prot_conc - mean_occ*tot_DNA_conc;
        if unbnd_conc < 1e-6 {
            unbnd_conc = 1e-6
        }
        (*unbnd_concs)[rnd] = unbnd_conc;
    }
}

fn calc_consensus_seq(bm: &BindingModel) -> Vec<Base> {
// Calculate the concensus sequence and it's maximum occupancy
    let mut consensus_seq = Vec::new();
    
    for score in bm.pos_gfes.iter() {
        let mut curr_base = Base::A;
        let mut curr_val = score.a;
        if  score.c > curr_val {
            curr_val = score.c;
            curr_base = Base::C;
        }
        if score.g > curr_val {
            curr_val = score.g;
            curr_base = Base::G;
        }
        if score.t > curr_val {
            // Dont need this
            //curr_val = score.t;
            curr_base = Base::T;
        }
        consensus_seq.push(curr_base)
    }
    
    consensus_seq
}


fn calc_lhd(
    obs_rds: &ObservedReads,
    bm: &BindingModel, 
    unbnd_concs: &Vec<f32>
        ) -> f32
{
    let mut lhd_num = 0.0; 
    let mut rnd_cnts = vec![0; obs_rds.max_rnd];
    for rd in obs_rds.rds.iter() {
        let unbnd_conc = unbnd_concs[(rd.round - 1) as usize];
        rnd_cnts[(rd.round - 1) as usize] += 1;
        let log_occ: f32 = calc_max_occupancy(
            &(rd.seq[..]), bm, unbnd_conc).ln();
        lhd_num += (rd.round as f32)*log_occ;
    }

    let norm_constants = estimate_average_occupancies(
        &obs_rds.rds, &bm, &unbnd_concs);
    print!("Norm Constants: {:?}\n", norm_constants);

    let num_distinct_seqs = (2.0 as f32).powi(bm.pos_gfes.len() as i32);
    let mut norm_constant = 0.0;
    for (constant, cnt) in norm_constants.iter().zip(rnd_cnts) {
        print!("{}\t{}\n", constant, cnt);
        norm_constant += constant*num_distinct_seqs*(cnt as f32);
    };
    print!("{}\t{}\n", lhd_num, norm_constant);
    lhd_num - norm_constant
}
 
fn main() {
    //// load the initial binidng model

    // 100-250 ng of soluble protein
    // 100 ng of selection ligand
    // 83 ng of non-sepcific ligand
    // all in 20 ul of h20 
    // Nucleotide mass: 5.0e-13 ug
    // => 6.5e12 total oligos
    // nG/L
    let tot_prot_conc = 200.0;
    let tot_DNA_conc = 150.0;
    
    let motif_fname = env::args().nth(1).unwrap();
    let mut bm = load_motif(&motif_fname);
    print!("{:?}\n", bm);

    let fastq_fname = env::args().nth(2).unwrap();
    let observed_reads = Box::new(load_reads_in_fastq(&fastq_fname, 1));

    let mut unbnd_concs = vec![tot_prot_conc/2.0; 
                               (observed_reads.max_rnd+1) as usize];
 
    update_unbnd_concs(
        &mut unbnd_concs, &observed_reads, &bm, tot_DNA_conc, tot_prot_conc);
    print!("Initial Unbnd Concs: {:?}\n", unbnd_concs);
    
    for i in 1..10 {
        bm.consensus_gfe *= 0.7;
        print!("Cons dG: {}\n", bm.consensus_gfe);
        update_unbnd_concs(
            &mut unbnd_concs, &observed_reads, &bm, tot_DNA_conc, tot_prot_conc);
        print!("Unbnd Concs: {:?}\n", unbnd_concs);

        let lhd = calc_lhd(
            &observed_reads, &bm, &unbnd_concs);
        print!("{}\t{}\tLhd: {}\n\n", i, unbnd_concs[0], lhd);
    }
    return;
}
