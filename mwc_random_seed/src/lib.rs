use std::num::Wrapping;

/// The multiply with carry rng.
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct MultiplyWithCarry {
    a: u32,
    c: Wrapping<u32>,
    x: Wrapping<u32>,
}

impl MultiplyWithCarry {
    pub fn new(a: u32, x: u32, c: u32) -> Self {
        Self {
            a,
            x: Wrapping(x),
            c: Wrapping(c),
        }
    }

    /// Access to the value part.
    pub fn value(&self) -> u32 {
        self.x.0
    }

    /// Access to the carry part.
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


    /// Generate a u32 that's below the limit.
    pub fn random_limited_u32(&mut self, limit: u32) -> u32 {
        if limit == 0 {
            return 0;
        }
        self.random_u32() % limit
    }
}

impl std::fmt::Debug for MultiplyWithCarry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[0x{:0>8x} 0x{:0>8x}]", self.c, self.x)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_mwc() {
        let mut rng = MultiplyWithCarry::new(1791398085, 1, 333 * 2);
        assert_eq!(rng.random_u32(), 0x6AC6935F);
        assert_eq!(rng.random_u32(), 0x2F2ED81B);
        assert_eq!(rng.random_u32(), 0x280687C4);
        assert_eq!(rng.random_u32(), 0xB6AAB839);
        assert_eq!(rng.random_u32(), 0xBFC793C3);
    }
}
