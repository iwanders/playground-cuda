use std::num::Wrapping;

/// Implements a 32 bit MWC; x(n)=a*x(n-1) + carry mod 2^32
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct MultiplyWithCarryCpu {
    /// The multiplication factor used by the algorithm.
    ///
    /// There are quite some options; 0xf83f630a is one used by opencv [1].
    /// There are more on the 1998 google groups archive [2]:
    ///   1791398085 1929682203 1683268614 1965537969 1675393560
    ///   1967773755 1517746329 1447497129 1655692410 1606218150
    ///   2051013963 1075433238 1557985959 1781943330 1893513180
    ///   1631296680 2131995753 2083801278 1873196400 1554115554
    /// [1] https://github.com/opencv/opencv/blob/0838920371bfa267c103890138553439b4a507e7/modules/core/include/opencv2/core/types_c.h#L216
    /// [2] https://groups.google.com/g/sci.math/c/ss3woKlsc3U/m/8K2TsYNAA_oJ
    a: u32,

    /// Current carry value.
    c: Wrapping<u32>,
    /// Current actual value.
    x: Wrapping<u32>,
}

impl MultiplyWithCarryCpu {
    pub fn new(a: u32, x: u32, c: u32) -> Self {
        Self {
            a,
            x: Wrapping(x),
            c: Wrapping(c),
        }
    }

    /// Access to the current value part.
    pub fn value(&self) -> u32 {
        self.x.0
    }

    /// Access to the current carry part.
    pub fn carry(&self) -> u32 {
        self.c.0
    }

    /// Advance the RNG and generate a new 32 bit integer
    pub fn random_u32(&mut self) -> u32 {
        let a: Wrapping<u64> = Wrapping(self.a as u64);
        let big_x = Wrapping(self.x.0 as u64);
        let big_carry = Wrapping(self.c.0 as u64);
        let new_value = a * big_x + big_carry;
        self.x = Wrapping(new_value.0 as u32);
        self.c = Wrapping((new_value.0 >> 32) as u32);
        self.x.0
    }


    /// Generate a u32 that's calculated modulo the limit.
    pub fn random_limited_u32(&mut self, limit: u32) -> u32 {
        if limit == 0 {
            return 0;
        }
        self.random_u32() % limit
    }
}

impl std::fmt::Debug for MultiplyWithCarryCpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[0x{:0>8x} 0x{:0>8x}]", self.c, self.x)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_mwc() {
        let mut rng = MultiplyWithCarryCpu::new(1791398085, 1, 333 * 2);
        assert_eq!(rng.random_u32(), 0x6AC6935F);
        assert_eq!(rng.random_u32(), 0x2F2ED81B);
        assert_eq!(rng.random_u32(), 0x280687C4);
        assert_eq!(rng.random_u32(), 0xB6AAB839);
        assert_eq!(rng.random_u32(), 0xBFC793C3);
    }
    #[test]
    fn test_mwc_find_seed() {
        let l = 150;
        let h = 500;
        let modulo = h - l;
        let max_advance = 500;
        let expected_values: [u32;7] = [201 - l, 484 - l, 188 - l, 496 - l, 432 - l, 347 - l, 356 - l];
        /*
           
            <-   n   ->
            *         *          *        *       *       *       *     *     main rng
                       \         drop      \
                        \                   \
                         \ inner rng1        \ inner rng2
            is n odd? even?
            inner rng gets initialised, first 'roll' is the value.

            <-   n   ->
            *         *          *        *         *       *       *     *     main rng
                       \          \        \         \
                        \          \        \         \
                         \ odd_in1  \        \ odd_in2 \
                                     \ even_in1         \ even_in2
            is n odd? even? 
            inner rng gets initialised, first 'roll' is the value.
        */
        for s in 1..=5 {
            let mut rng = MultiplyWithCarryCpu::new(1791398085, s, 333 * 2);
            // rng.random_u32();
            let mut past_values = [[0; 7];2];
            for advance in 0..max_advance {
                let inner_init = rng.random_u32();

                let mut inner_rng = MultiplyWithCarryCpu::new(1791398085, inner_init, 333 * 2);
                let new_value = inner_rng.random_limited_u32(modulo);
                let this_entry = &mut past_values[(advance % 2) as usize];
                this_entry.rotate_left(1);
                this_entry[expected_values.len() - 1] = new_value;
                if *this_entry == expected_values {
                    println!("Found it at {advance}");
                    return;
                }
                // println!("past_values: {past_values:?}");
            }
        }
        assert!(false, "if we got here we didn't find the seed");
    }

    #[test]
    fn test_mwc_check_modulo() {
        let l = 150;
        let h = 500;
        let modulo = h - l;
        let max_advance = 500;
        let max: u64 = 1 << 32;
        let search = 460 - l;
        let mut found_seeds = vec![];
        for s in 0..=max {
            let mut inner_rng = MultiplyWithCarryCpu::new(1791398085, s as u32, 333 * 2);
            let value = inner_rng.random_u32() % modulo;
            if value == search {
                found_seeds.push(s as u32);
            }
            
        }
        println!("Found {} seeds that match {search}", found_seeds.len());
    }


    // http://jmasm.com/index.php/jmasm/article/view/57/56
    // https://digitalcommons.wayne.edu/jmasm/vol2/iss1/2/
    // DOI 10.22237/jmasm/1051747320
    // page 7;
    #[test]
    fn is_it_an_lcg_with_modulo(){
        // And still another feature of the MWC sequence generated on pairs [c, x] by means of
        // f([a, c]) = [_(ax + c)_, (ax + c) mod b] is that the resultsing x's are just the
        // elements of the congruential sequence y_n = ay_n-1 mod (ab - 1) reduced mod b.
        // with seed y0 is s * b + c
        // reproduce paper
        {
            let a = 698769069;
            let c = 123;
            let x = 456789;
            let mut rng = MultiplyWithCarryCpu::new(a, x, c);
            let b : u128 = 1 << 32;
            let y0: u128 = c as u128 * b + x as u128;
            let mut yn: u128 = y0;
            for i in 0..10 {
                let mwc = rng.random_u32();
                yn = ((a as u128) * yn) % (((a as u128) * b) -1);
                let lcg = yn % (1<<32) ;
                println!("mwc: {mwc} lcg: {lcg}");
                assert_eq!(mwc, lcg as u32);
            }
        }
        for s in [1, 1337, 354322, 1702494920] {
            let a = 1791398085;
            let c = 333 * 2;
            let x = s;
            let mut rng = MultiplyWithCarryCpu::new(a, x, c);
            let b : u128 = 1 << 32;
            let y0: u128 = c as u128 * b + x as u128;
            let mut yn: u128 = y0;
            for i in 0..100 {
                let mwc = rng.random_u32();
                yn = ((a as u128) * yn) % (((a as u128) * b) -1);
                let lcg = yn % (1<<32) ;
                // println!("mwc: {mwc} lcg: {lcg}");
                assert_eq!(mwc, lcg as u32);
            }
        }
    }
}
