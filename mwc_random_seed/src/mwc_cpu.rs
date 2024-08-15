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
}
