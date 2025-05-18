// ModelMetadata.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ModelMetadata {
    struct Model {
        string title;
        string username;
        string accountId;
        uint256 timestamp;
    }

    mapping(string => Model) public models;

    function storeModel(string memory _hashId, string memory _title, string memory _username, string memory _accountId) public {
        models[_hashId] = Model(_title, _username, _accountId, block.timestamp);
    }

    function getModel(string memory _hashId) public view returns (string memory, string memory, string memory, uint256) {
        Model memory m = models[_hashId];
        return (m.title, m.username, m.accountId, m.timestamp);
    }
}
