# Ideas
* Instead of permuting the scales individually, multiply unpermuted scales, then permute the result  (doesn't seem to improve performance).
* Get rid of masked loads so that the compiler can use `vperm*` directly on memory operands (promising).
* Somehow use one `vpdpbusds` instead of two (doesn't seem to be possible).
* Somehow use the accumulator in `vpdpbusds` instead of a separate subtraction at the very end (also doesn't seem to be possible).
