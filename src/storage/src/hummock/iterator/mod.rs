// Copyright 2022 Singularity Data
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::{HummockResult, HummockValue};

mod concat;
pub use concat::*;
mod concat_inner;
mod reverse_concat;
pub use reverse_concat::*;
mod reverse_merge;
pub use reverse_merge::*;
mod reverse_user;
pub use reverse_user::*;
mod merge;
pub use merge::*;
mod merge_inner;
pub mod user;
pub use user::*;

#[cfg(test)]
pub(crate) mod test_utils;

use async_trait::async_trait;

/// `HummockIterator` defines the interface of all iterators, including `SSTableIterator`,
/// `MergeIterator`, `UserIterator` and `ConcatIterator`.
///
/// After creating the iterator instance,
/// - if you want to iterate from the beginning, you need to then call its `rewind` method.
/// - if you want to iterate from some specific position, you need to then call its `seek` method.
#[async_trait]
pub trait HummockIterator: Send + Sync {
    type Direction: HummockIteratorDirection;

    /// Moves a valid iterator to the next key.
    ///
    /// Note:
    /// - Before calling this function, makes sure the iterator `is_valid`.
    /// - After calling this function, you may first check whether the iterator `is_valid` again,
    ///   then get the new data by calling `key` and `value`.
    /// - If the position after calling this is invalid, this function WON'T return an `Err`. You
    ///   should check `is_valid` before continuing the iteration.
    ///
    /// # Panics
    /// This function will panic if the iterator is invalid.
    async fn next(&mut self) -> HummockResult<()>;

    /// Retrieves the current key.
    ///
    /// Note:
    /// - Before calling this function, makes sure the iterator `is_valid`.
    /// - This function should be straightforward and return immediately.
    ///
    /// # Panics
    /// This function will panic if the iterator is invalid.
    // TODO: Add lifetime
    fn key(&self) -> &[u8];

    /// Retrieves the current value, decoded as [`HummockValue`].
    ///
    /// Note:
    /// - Before calling this function, makes sure the iterator `is_valid`.
    /// - This function should be straightforward and return immediately.
    ///
    /// # Panics
    /// This function will panic if the iterator is invalid, or the value cannot be decoded into
    /// [`HummockValue`].
    // TODO: Add lifetime
    fn value(&self) -> HummockValue<&[u8]>;

    /// Indicates whether the iterator can be used.
    ///
    /// Note:
    /// - ONLY call `key`, `value`, and `next` if `is_valid` returns `true`.
    /// - This function should be straightforward and return immediately.
    fn is_valid(&self) -> bool;

    /// Resets the position of the iterator.
    ///
    /// Note:
    /// - Do not decide whether the position is valid or not by checking the returned error of this
    ///   function. This function WON'T return an `Err` if invalid. You should check `is_valid`
    ///   before starting iteration.
    async fn rewind(&mut self) -> HummockResult<()>;

    /// Resets iterator and seeks to the first position where the key >= provided key, or key <=
    /// provided key if this is a reverse iterator.
    ///
    /// Note:
    /// - Do not decide whether the position is valid or not by checking the returned error of this
    ///   function. This function WON'T return an `Err` if invalid. You should check `is_valid`
    ///   before starting iteration.
    async fn seek(&mut self, key: &[u8]) -> HummockResult<()>;
}

pub trait ForwardHummockIterator = HummockIterator<Direction = Forward>;
pub trait BackwardHummockIterator = HummockIterator<Direction = Backward>;

pub type BoxedForwardHummockIterator = Box<dyn ForwardHummockIterator>;
pub type BoxedBackwardHummockIterator = Box<dyn BackwardHummockIterator>;
pub type BoxedHummockIterator<D> = Box<dyn HummockIterator<Direction = D>>;

pub enum DirectionEnum {
    Forward,
    Backward,
}

pub trait HummockIteratorDirection: Sync + Send {
    fn direction() -> DirectionEnum;
}

pub struct Forward;
impl HummockIteratorDirection for Forward {
    #[inline(always)]
    fn direction() -> DirectionEnum {
        DirectionEnum::Forward
    }
}

pub struct Backward;
impl HummockIteratorDirection for Backward {
    #[inline(always)]
    fn direction() -> DirectionEnum {
        DirectionEnum::Backward
    }
}
