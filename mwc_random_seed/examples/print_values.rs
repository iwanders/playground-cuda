use clap::Parser;
use mwc_random_seed;

/// A program to output values from a 32 bit MWC algorithm; x(n)=a*x(n-1) + carry mod 2^32
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The multiplication factor.
    ///
    /// There are quite some options; 0xf83f630a is one used by opencv [1].
    /// There are more on the 1998 google groups archive [2]:
    ///   1791398085 1929682203 1683268614 1965537969 1675393560
    ///   1967773755 1517746329 1447497129 1655692410 1606218150
    ///   2051013963 1075433238 1557985959 1781943330 1893513180
    ///   1631296680 2131995753 2083801278 1873196400 1554115554
    /// [1] https://github.com/opencv/opencv/blob/0838920371bfa267c103890138553439b4a507e7/modules/core/include/opencv2/core/types_c.h#L216
    /// [2] https://groups.google.com/g/sci.math/c/ss3woKlsc3U/m/8K2TsYNAA_oJ
    factor: u32,

    /// The start value for the actualy value of x.
    value: u32,

    /// The start value for the carry value.
    carry: u32,

    /// Number of advances to perform.
    #[arg(short, long, default_value_t = 5)]
    advances: usize,
}

fn main() {
    let args = Args::parse();
    let mut rng = mwc_random_seed::MultiplyWithCarryCpu::new(args.factor, args.value, args.carry);

    for _ in 0..args.advances {
        println!("{:0>8X}", rng.random_u32());
    }
}
